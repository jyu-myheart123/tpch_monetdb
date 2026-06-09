import logging
import os
import subprocess
import shlex
import signal
import time
from pathlib import Path
import select


logger = logging.getLogger(__name__)


class RunnerTransportError(Exception):
    """Raised when the runner stdin pipe is broken or unreadable."""
    pass


class RunnerBrokenPipePersistentError(Exception):
    """Raised when runner transport recovery fails after one replay attempt."""
    pass


class RunnerInfraFailureError(Exception):
    def __init__(self, failure_code: str, detail: str) -> None:
        self.failure_code = failure_code
        self.detail = detail
        super().__init__(f"[ERROR:{failure_code}] {detail}")
        return None


class FasttestProc:
    """Persistent db process wrapper used by TPC-H MonetDB test and profiling tools."""

    def __init__(
        self,
        command: str,
        *,
        echo_output: bool = False,
        cwd: Path,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._echo_output = echo_output
        self._cwd = cwd
        self._extra_env = dict(extra_env or {})
        self._proc: subprocess.Popen[bytes] | None = None
        self._p2c_w: int | None = None
        self._c2p_file = None
        self._c2p_r: int | None = None
        self._stdout_fd: int | None = None
        self._stderr_fd: int | None = None
        self._stdin = None

    @property
    def pid(self) -> int | None:
        """Return the child process PID after the runner has started."""
        if self._proc is None:
            return None
        return self._proc.pid

    def start_for_external_control(self) -> int:
        """Start the runner and expose its PID for external profilers."""
        self._start()
        if self._proc is None or self._proc.pid is None:
            raise RuntimeError("runner failed to start")
        return self._proc.pid

    def _start(self) -> None:
        if self._proc is not None:
            return
        p2c_r, p2c_w = os.pipe()
        c2p_r, c2p_w = os.pipe()

        if isinstance(self._command, str):
            cmd = self._command.strip()
            cmd = cmd if cmd else "./db"
            argv = shlex.split(cmd)
            if not argv:
                argv = ["./db"]
        else:
            argv = [str(self._command)]
        self._proc = subprocess.Popen(
            argv,
            pass_fds=(p2c_r, c2p_w),
            env={
                **os.environ,
                **self._extra_env,
                "P2C_FD": str(p2c_r),
                "C2P_FD": str(c2p_w),
            },
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
        )

        os.close(p2c_r)
        os.close(c2p_w)

        self._p2c_w = p2c_w
        self._c2p_r = c2p_r
        os.set_blocking(c2p_r, False)
        self._c2p_file = os.fdopen(c2p_r, "rb", buffering=0)
        self._stdin = self._proc.stdin
        if self._proc.stdout is not None:
            self._stdout_fd = self._proc.stdout.fileno()
            os.set_blocking(self._stdout_fd, False)
        if self._proc.stderr is not None:
            self._stderr_fd = self._proc.stderr.fileno()
            os.set_blocking(self._stderr_fd, False)

    def _process_pid(self) -> int | None:
        """Return the live subprocess pid when the Popen object exposes one."""
        if self._proc is None:
            return None
        pid = getattr(self._proc, "pid", None)
        if isinstance(pid, int) and pid > 0:
            return pid
        return None

    def _process_group_id(self) -> int | None:
        """Return the runner process-group id created by start_new_session."""
        pid = self._process_pid()
        if pid is None:
            return None
        try:
            return os.getpgid(pid)
        except OSError:
            return pid

    def _signal_process_tree(self, sig: int) -> None:
        """Signal the whole runner process group, falling back to the root pid."""
        pgid = self._process_group_id()
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return None
            except OSError:
                pass
        pid = self._process_pid()
        if pid is not None:
            try:
                os.kill(pid, sig)
                return None
            except OSError:
                pass
        if self._proc is not None and sig == signal.SIGKILL:
            kill = getattr(self._proc, "kill", None)
            if callable(kill):
                kill()
        return None

    def _wait_for_exit(
        self,
        proc: subprocess.Popen[bytes],
        *,
        timeout: int,
    ) -> bool:
        """Wait for the runner process and report whether it exited in time."""
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _decode_buffers(
        self,
        resp_buf: bytearray,
        out_buf: bytearray,
        err_buf: bytearray,
    ) -> tuple[str, str, str]:
        """Decode accumulated response, stdout, and stderr buffers."""
        return (
            resp_buf.decode("utf-8", errors="replace"),
            out_buf.decode("utf-8", errors="replace"),
            err_buf.decode("utf-8", errors="replace"),
        )

    def _timeout_result(
        self,
        *,
        timeout: int,
        resp_buf: bytearray,
        out_buf: bytearray,
        err_buf: bytearray,
    ) -> tuple[str, str, str]:
        """Terminate the runner after a response timeout and return diagnostics."""
        response, out, err = self._decode_buffers(resp_buf, out_buf, err_buf)
        drain_out, drain_err = self.terminate_and_drain(suppress_errors=True)
        timeout_message = f"Terminated after {timeout} seconds due to timeout."
        response = "\n".join(part for part in (response, timeout_message) if part)
        return response, out + drain_out, err + drain_err

    def _write_fd_all(self, fd: int, data: bytes) -> None:
        offset = 0
        try:
            while offset < len(data):
                written = os.write(fd, data[offset:])
                if written <= 0:
                    raise RunnerTransportError("Runner control pipe wrote zero bytes")
                offset += written
        except BrokenPipeError as exc:
            raise RunnerTransportError(
                "Runner control pipe is broken (BrokenPipeError)"
            ) from exc
        except OSError as exc:
            if exc.errno in (9, 32):
                raise RunnerTransportError(
                    f"Runner control pipe is closed or unreadable (OSError {exc.errno})"
                ) from exc
            raise
        return None

    def _write_run_control(self) -> None:
        if self._p2c_w is None or self._c2p_file is None or self._c2p_r is None:
            raise RuntimeError("runner not initialized")
        self._write_fd_all(self._p2c_w, b"run\n")
        return None

    def _read_run_response(self, timeout: int = 0) -> tuple[str, str, str]:
        """Read one control response while draining child stdout and stderr."""
        out_buf = bytearray()
        err_buf = bytearray()
        resp_buf = bytearray()
        deadline = time.monotonic() + timeout if timeout > 0 else None

        while True:
            if self._c2p_r is None:
                raise RunnerTransportError("Runner response pipe is closed")
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._timeout_result(
                        timeout=timeout,
                        resp_buf=resp_buf,
                        out_buf=out_buf,
                        err_buf=err_buf,
                    )
                select_timeout = min(1.0, remaining)
            else:
                select_timeout = None

            fds = [self._c2p_r]
            if self._stdout_fd is not None:
                fds.append(self._stdout_fd)
            if self._stderr_fd is not None:
                fds.append(self._stderr_fd)

            rlist, _, _ = select.select(fds, [], [], select_timeout)
            if not rlist and deadline is not None:
                continue

            for fd in rlist:
                if fd == self._c2p_r:
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        rc = self._proc.wait() if self._proc is not None else None
                        if rc is not None:
                            err_buf.extend(
                                f"process exited with code {rc}\n".encode("utf-8")
                            )
                        while self._stdout_fd is not None:
                            more = os.read(self._stdout_fd, 4096)
                            if not more:
                                break
                            out_buf.extend(more)
                        while self._stderr_fd is not None:
                            more = os.read(self._stderr_fd, 4096)
                            if not more:
                                break
                            err_buf.extend(more)
                        response, out, err = self._decode_buffers(
                            resp_buf,
                            out_buf,
                            err_buf,
                        )
                        return response, out, err
                    resp_buf.extend(chunk)
                elif fd == self._stdout_fd:
                    chunk = os.read(fd, 4096)
                    if chunk:
                        out_buf.extend(chunk)
                        if self._echo_output:
                            os.write(1, chunk)
                elif fd == self._stderr_fd:
                    chunk = os.read(fd, 4096)
                    if chunk:
                        err_buf.extend(chunk)
                        if self._echo_output:
                            os.write(2, chunk)
            if b"\n" in resp_buf:
                line, _, rest = resp_buf.partition(b"\n")
                response = line.decode("utf-8", errors="replace")
                out = out_buf.decode("utf-8", errors="replace")
                err = err_buf.decode("utf-8", errors="replace")
                return response, out, err

    def run(self, timeout: int = 0) -> tuple[str, str, str]:
        self._start()
        self._write_run_control()
        return self._read_run_response(timeout)

    def run_batch(
        self,
        args_list: list[str],
        timeout: int = 0,
    ) -> tuple[str, str, str]:
        self._start()
        for args in args_list:
            self.send(args)
        self.send("")
        self._write_run_control()
        return self._read_run_response(timeout)

    def send(self, line: str) -> None:
        self._start()
        if self._stdin is None:
            raise RuntimeError("stdin not available")
        try:
            self._stdin.write((line + "\n").encode("utf-8"))
            self._stdin.flush()
        except BrokenPipeError as exc:
            raise RunnerTransportError(
                "Runner stdin pipe is broken (BrokenPipeError)"
            ) from exc
        except OSError as exc:
            if exc.errno in (9, 32):  # EBADF, EPIPE
                raise RunnerTransportError(
                    f"Runner stdin pipe is closed or unreadable (OSError {exc.errno})"
                ) from exc
            raise
        except ValueError as exc:
            if "write to closed file" in str(exc):
                raise RunnerTransportError(
                    "Runner stdin is closed (ValueError)"
                ) from exc
            raise

    def close_stdin(self) -> None:
        if self._stdin is not None:
            self._stdin.close()
            self._stdin = None

    def _close_transport_fds(self) -> None:
        if self._p2c_w is not None:
            try:
                os.close(self._p2c_w)
            except OSError:
                pass
            self._p2c_w = None
        if self._c2p_file is not None:
            try:
                self._c2p_file.close()
            except Exception:
                pass
            self._c2p_file = None
        self._c2p_r = None
        if self._stdin is not None:
            try:
                self._stdin.close()
            except Exception:
                pass
            self._stdin = None

    def terminate(self, *, suppress_errors: bool = False) -> None:
        self.terminate_and_drain(suppress_errors=suppress_errors)
        return None

    def terminate_and_drain(
        self,
        *,
        suppress_errors: bool = False,
    ) -> tuple[str, str]:
        if self._proc is None:
            return "", ""
        proc = self._proc
        if self._p2c_w is not None:
            try:
                self._write_fd_all(self._p2c_w, b"stop\n")
            except Exception:
                pass
        self._close_transport_fds()
        exited = self._wait_for_exit(proc, timeout=5)
        if not exited:
            self._signal_process_tree(signal.SIGTERM)
            exited = self._wait_for_exit(proc, timeout=2)
        if not exited:
            self._signal_process_tree(signal.SIGKILL)
            exited = self._wait_for_exit(proc, timeout=5)
        if not exited and not suppress_errors:
            raise RuntimeError("process did not exit after SIGKILL")
        out = self._drain_fd(self._stdout_fd)
        err = self._drain_fd(self._stderr_fd)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
        if proc.returncode not in (0, None) and not suppress_errors:
            raise RuntimeError(f"process exited with code {proc.returncode}")
        self._proc = None
        self._stdout_fd = None
        self._stderr_fd = None
        return (
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
        )

    def _drain_fd(self, fd: int | None) -> bytes:
        if fd is None:
            return b""
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                break
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
