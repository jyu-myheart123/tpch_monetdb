import os
import asyncio
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-only sandbox (Landlock)"
)
MACOS_ONLY = pytest.mark.skipif(
    sys.platform != "darwin", reason="macOS-only sandbox tests"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _have_landlock() -> bool:
    try:
        import landlock  # noqa: F401

        return True
    except Exception:
        return False


def _sandbox_available_or_skip() -> None:
    """
    Skip if landlock isn't installed or Landlock can't be applied on this kernel.
    We probe by launching a tiny process that applies the sandbox and exits.
    """
    if not _have_landlock():
        pytest.skip("landlock not installed")

    probe = textwrap.dedent(
        """
        import sys
        from tpch_monetdb.tools.sandbox import SandboxConfig, _apply_sandbox

        d = sys.argv[1]
        _apply_sandbox(SandboxConfig(writable_roots=[d]).normalized())
        sys.exit(0)
        """
    ).strip()

    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, "-c", probe, d], capture_output=True, text=True
        )
    if r.returncode != 0:
        pytest.skip(
            f"Landlock sandbox not supported/enabled here: {r.stderr.strip() or r.stdout.strip()}"
        )


@pytest.fixture
def landlock_required() -> None:
    if sys.platform == "linux":
        _sandbox_available_or_skip()
    return None


@pytest.fixture
def rw_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def other_dir() -> str:
    with tempfile.TemporaryDirectory() as d:
        yield d


@LINUX_ONLY
def test_popen_allows_write_in_writable_roots_and_denies_elsewhere(
    landlock_required, rw_dir, other_dir
):
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_popen

    allowed = os.path.join(rw_dir, "ok.txt")
    denied = os.path.join(other_dir, "no.txt")

    code = textwrap.dedent(
        f"""
        import sys, os
        def w(p):
            try:
                with open(p, "wb") as f:
                    f.write(b"x")
                return True
            except OSError as e:
                print("FAIL", p, getattr(e, "errno", None), type(e).__name__)
                return False

        a = w({allowed!r})
        b = w({denied!r})
        sys.exit(0 if (a and not b) else 2)
        """
    ).strip()

    p = sandbox_popen(
        [sys.executable, "-c", code],
        cfg=SandboxConfig(
            writable_roots=[rw_dir], cwd=rw_dir, cpu_seconds=2, nproc=None
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(timeout=10)

    assert p.returncode == 0, f"stdout:\n{out}\n\nstderr:\n{err}"
    assert os.path.exists(allowed)
    assert not os.path.exists(denied)


@LINUX_ONLY
def test_popen_denies_tmp_write_by_default(landlock_required, rw_dir):
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_popen

    code = textwrap.dedent(
        """
        import os, sys
        p = "/tmp/landlock_tmp_should_fail.txt"
        try:
            with open(p, "wb") as f:
                f.write(b"nope")
            print("UNEXPECTED_OK", p)
            sys.exit(3)
        except OSError as e:
            print("EXPECTED_FAIL", getattr(e, "errno", None), type(e).__name__)
            sys.exit(0)
        """
    ).strip()

    p = sandbox_popen(
        [sys.executable, "-c", code],
        cfg=SandboxConfig(
            writable_roots=[rw_dir], cwd=rw_dir, cpu_seconds=2, nproc=None
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(timeout=10)
    assert p.returncode == 0, f"stdout:\n{out}\n\nstderr:\n{err}"


@pytest.mark.asyncio
@LINUX_ONLY
async def test_async_shell_allows_write_in_workspace_and_denies_tmp(
    landlock_required, rw_dir
):
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_shell_async

    cfg = SandboxConfig(writable_roots=[rw_dir], cwd=rw_dir, cpu_seconds=2, nproc=None)

    proc = await sandbox_shell_async(
        "echo hi > ok.txt; echo nope > /tmp/landlock_should_fail_async.txt",
        cfg=cfg,
    )
    out, err = await asyncio.wait_for(proc.communicate(), timeout=10)

    # The shell will return nonzero because the second redirect should fail.
    assert proc.returncode != 0

    # Verify the allowed write happened.
    assert os.path.exists(os.path.join(rw_dir, "ok.txt"))
    # Verify /tmp write did not happen (best-effort: file shouldn't exist)
    assert not os.path.exists("/tmp/landlock_should_fail_async.txt")

    # Helpful debug on failure
    if err:
        _ = err.decode(errors="replace")


@pytest.mark.asyncio
@LINUX_ONLY
async def test_async_exec_spawning_child_inherits_restrictions(
    landlock_required, rw_dir, other_dir
):
    """
    Verify that forking/spawning a subprocess doesn't break out: the child is still restricted.
    We run a python that spawns another python, which tries a denied write.
    """
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_exec_async

    denied = os.path.join(other_dir, "nope.txt")

    inner = textwrap.dedent(
        f"""
        import sys
        try:
            open({denied!r}, "wb").write(b"x")
            print("INNER_UNEXPECTED_OK")
            sys.exit(5)
        except OSError as e:
            print("INNER_EXPECTED_FAIL", getattr(e, "errno", None), type(e).__name__)
            sys.exit(0)
        """
    ).strip()

    outer = textwrap.dedent(
        f"""
        import subprocess, sys, textwrap
        inner = {inner!r}
        r = subprocess.run([sys.executable, "-c", inner], capture_output=True, text=True)
        print("INNER_RC", r.returncode)
        print(r.stdout)
        print(r.stderr)
        sys.exit(0 if r.returncode == 0 else 6)
        """
    ).strip()

    cfg = SandboxConfig(writable_roots=[rw_dir], cwd=rw_dir, cpu_seconds=5, nproc=None)

    proc = await sandbox_exec_async(sys.executable, "-c", outer, cfg=cfg)
    out, err = await asyncio.wait_for(proc.communicate(), timeout=15)

    assert proc.returncode == 0, (
        f"stdout:\n{(out or b'').decode(errors='replace')}\n\nstderr:\n{(err or b'').decode(errors='replace')}"
    )
    assert not os.path.exists(denied)


@LINUX_ONLY
def test_no_new_privs_is_set(landlock_required, rw_dir):
    """
    Verify PR_SET_NO_NEW_PRIVS is in effect inside the sandboxed process.
    We check /proc/self/status contains NoNewPrivs: 1
    """
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_popen

    code = textwrap.dedent(
        """
        import sys
        s = open("/proc/self/status", "r", encoding="utf-8", errors="replace").read()
        # line looks like: "NoNewPrivs:\t1"
        ok = any(line.startswith("NoNewPrivs:") and line.strip().endswith("1") for line in s.splitlines())
        sys.exit(0 if ok else 9)
        """
    ).strip()

    p = sandbox_popen(
        [sys.executable, "-c", code],
        cfg=SandboxConfig(
            writable_roots=[rw_dir], cwd=rw_dir, cpu_seconds=2, nproc=None
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, err = p.communicate(timeout=10)
    assert p.returncode == 0, f"stdout:\n{out}\n\nstderr:\n{err}"


def test_normalized_merges_legacy_roots_and_explicit_rules(tmp_path) -> None:
    from tpch_monetdb.tools.sandbox import SandboxConfig

    allow_path = tmp_path / "allow"
    deny_path = tmp_path / "deny"
    read_path = tmp_path / "read.txt"
    tmp_root = tmp_path / "tmp"
    allow_path.mkdir()
    deny_path.mkdir()
    read_path.write_text("x", encoding="utf-8")
    tmp_root.mkdir()

    cfg = SandboxConfig(
        writable_roots=[str(allow_path)],
        allow_write=[str(allow_path)],
        deny_write=[str(deny_path)],
        deny_read=[str(read_path)],
        allow_read=[str(read_path)],
        tmp_root=str(tmp_root),
    ).normalized()

    assert str(allow_path.resolve()) in cfg.allow_write
    assert str(tmp_root.resolve()) in cfg.allow_write
    assert str(allow_path.resolve()) in cfg.writable_roots
    assert str(deny_path.resolve()) in cfg.deny_write
    assert str(read_path.resolve()) in cfg.deny_read
    assert str(read_path.resolve()) in cfg.allow_read


def test_unavailable_reason_reports_missing_macos_dependency(monkeypatch) -> None:
    from tpch_monetdb.tools import sandbox as sandbox_module
    from tpch_monetdb.tools.sandbox import SandboxConfig

    monkeypatch.setattr(sandbox_module, "_IS_LINUX", False)
    monkeypatch.setattr(sandbox_module, "_IS_MACOS", True)
    monkeypatch.setattr(
        sandbox_module,
        "_check_macos_sandbox_dependencies",
        lambda: ["sandbox-exec not found"],
    )

    reason = sandbox_module._get_sandbox_unavailable_reason(SandboxConfig())

    assert reason == "sandbox-exec not found"


@pytest.mark.asyncio
@MACOS_ONLY
async def test_macos_tmpdir_is_injected_and_writable(tmp_path) -> None:
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_exec_async

    workdir = tmp_path / "work"
    tmp_root = tmp_path / "sandbox-tmp"
    workdir.mkdir()
    tmp_root.mkdir()
    code = textwrap.dedent(
        """
        import os
        import tempfile

        fd, path = tempfile.mkstemp()
        os.write(fd, b"x")
        os.close(fd)
        print(os.environ["TMPDIR"])
        print(path)
        """
    ).strip()

    cfg = SandboxConfig(
        writable_roots=[str(workdir)],
        cwd=str(workdir),
        tmp_root=str(tmp_root),
        fail_if_unavailable=True,
        nproc=None,
    )
    proc = await sandbox_exec_async(
        sys.executable,
        "-c",
        code,
        cfg=cfg,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    assert proc.returncode == 0, f"stdout:\n{stdout_text}\n\nstderr:\n{stderr_text}"
    lines = [line for line in stdout_text.strip().splitlines() if line]
    assert lines[0] == str(tmp_root.resolve())
    assert lines[1].startswith(str(tmp_root.resolve()))


@pytest.mark.asyncio
@MACOS_ONLY
async def test_macos_deny_write_overrides_broader_allow(tmp_path) -> None:
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_shell_async

    workdir = tmp_path / "work"
    blocked = workdir / "blocked"
    workdir.mkdir()
    blocked.mkdir()
    cfg = SandboxConfig(
        writable_roots=[str(workdir)],
        deny_write=[str(blocked)],
        cwd=str(workdir),
        fail_if_unavailable=True,
        nproc=None,
    )
    proc = await sandbox_shell_async(
        "echo ok > ok.txt; echo no > blocked/no.txt",
        cfg=cfg,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)

    assert proc.returncode != 0, (
        f"stdout:\n{stdout.decode('utf-8', errors='replace')}\n\n"
        f"stderr:\n{stderr.decode('utf-8', errors='replace')}"
    )
    assert (workdir / "ok.txt").exists()
    assert not (blocked / "no.txt").exists()


@MACOS_ONLY
def test_macos_fail_if_unavailable_modes(monkeypatch, tmp_path) -> None:
    from tpch_monetdb.tools import sandbox as sandbox_module
    from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_popen

    monkeypatch.setattr(
        sandbox_module,
        "_check_macos_sandbox_dependencies",
        lambda: ["sandbox-exec not found"],
    )

    strict_cfg = SandboxConfig(
        writable_roots=[str(tmp_path)],
        cwd=str(tmp_path),
        fail_if_unavailable=True,
    )
    with pytest.raises(RuntimeError, match="sandbox-exec not found"):
        sandbox_popen(
            [sys.executable, "-c", "print('strict')"],
            cfg=strict_cfg,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    relaxed_cfg = SandboxConfig(
        writable_roots=[str(tmp_path)],
        cwd=str(tmp_path),
        fail_if_unavailable=False,
    )
    proc = sandbox_popen(
        [sys.executable, "-c", "print('relaxed')"],
        cfg=relaxed_cfg,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
    assert "relaxed" in stdout
