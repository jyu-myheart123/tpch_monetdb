from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

_LOGGER = logging.getLogger(__name__)

_IS_LINUX = platform.system() == "Linux"
_IS_MACOS = platform.system() == "Darwin"

if _IS_LINUX:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    PR_SET_NO_NEW_PRIVS = 38
else:
    libc = None
    PR_SET_NO_NEW_PRIVS = 38


def _normalize_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _dedupe_paths(paths: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_path in paths:
        path = _normalize_path(raw_path)
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return tuple(normalized)


def _escape_seatbelt_path(path: str) -> str:
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return escaped


def _seatbelt_path_rules(effect: str, operation: str, path: str) -> list[str]:
    escaped = _escape_seatbelt_path(path)
    return [
        f'({effect} {operation} (literal "{escaped}"))',
        f'({effect} {operation} (subpath "{escaped}"))',
    ]


def _prctl_set_no_new_privs() -> None:
    if not _IS_LINUX or libc is None:
        return None
    rc = libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if rc != 0:
        error_num = ctypes.get_errno()
        raise OSError(
            error_num,
            f"prctl(PR_SET_NO_NEW_PRIVS) failed: {os.strerror(error_num)}",
        )
    return None


def _set_rlimits(
    *,
    cpu_seconds: int | None = 10,
    as_bytes: int | None = 512 * 1024 * 1024,
    fsize_bytes: int | None = 50 * 1024 * 1024,
    nofile: int | None = 256,
    nproc: int | None = 128,
) -> None:
    def _set_limit(
        limit: int,
        value: int | None,
        *,
        ignore_failure: bool = False,
    ) -> None:
        if value is None:
            return None
        try:
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError):
            if ignore_failure:
                return None
            raise
        return None

    _set_limit(resource.RLIMIT_CPU, cpu_seconds)
    _set_limit(resource.RLIMIT_AS, as_bytes, ignore_failure=_IS_MACOS)
    _set_limit(resource.RLIMIT_FSIZE, fsize_bytes)
    _set_limit(resource.RLIMIT_NOFILE, nofile)
    _set_limit(resource.RLIMIT_NPROC, nproc, ignore_failure=_IS_MACOS)
    return None


@dataclass(frozen=True)
class SandboxConfig:
    writable_roots: Sequence[str] = ()
    allow_write: Sequence[str] = ()
    deny_write: Sequence[str] = ()
    deny_read: Sequence[str] = ()
    allow_read: Sequence[str] = ()
    cwd: str | None = None
    tmp_root: str | None = None
    fail_if_unavailable: bool = False
    cpu_seconds: int | None = 10
    as_bytes: int | None = 512 * 1024 * 1024
    fsize_bytes: int | None = 50 * 1024 * 1024
    nofile: int | None = 256
    nproc: int | None = 128
    umask: int = 0o077

    def normalized(self) -> "SandboxConfig":
        merged_allow_write = list(self.allow_write)
        merged_allow_write.extend(self.writable_roots)
        if self.tmp_root is not None:
            merged_allow_write.append(self.tmp_root)
        allow_write = _dedupe_paths(merged_allow_write)
        deny_write = _dedupe_paths(self.deny_write)
        deny_read = _dedupe_paths(self.deny_read)
        allow_read = _dedupe_paths(self.allow_read)
        cwd = _normalize_path(self.cwd) if self.cwd else None
        tmp_root = _normalize_path(self.tmp_root) if self.tmp_root else None
        if tmp_root is not None and tmp_root not in allow_write:
            allow_write = _dedupe_paths([*allow_write, tmp_root])
        return SandboxConfig(
            writable_roots=allow_write,
            allow_write=allow_write,
            deny_write=deny_write,
            deny_read=deny_read,
            allow_read=allow_read,
            cwd=cwd,
            tmp_root=tmp_root,
            fail_if_unavailable=self.fail_if_unavailable,
            cpu_seconds=self.cpu_seconds,
            as_bytes=self.as_bytes,
            fsize_bytes=self.fsize_bytes,
            nofile=self.nofile,
            nproc=self.nproc,
            umask=self.umask,
        )


def _apply_sandbox_linux(cfg: SandboxConfig) -> None:
    """Apply the Linux Landlock sandbox and process limits in the child."""
    if sys.platform != "linux":
        raise RuntimeError("This sandbox is Linux-only")

    from landlock import Ruleset

    _prctl_set_no_new_privs()
    _set_rlimits(
        cpu_seconds=cfg.cpu_seconds,
        as_bytes=cfg.as_bytes,
        fsize_bytes=cfg.fsize_bytes,
        nofile=cfg.nofile,
        nproc=cfg.nproc,
    )

    rs = Ruleset()
    if hasattr(rs, "handle_write"):
        rs.handle_write()
    elif hasattr(rs, "restrict_writes"):
        rs.restrict_writes()
    else:
        write_access = None
        for name in ("AccessFS", "FSAccess", "Access", "FS"):
            if hasattr(__import__("landlock"), name):
                write_access = getattr(__import__("landlock"), name)
                break
        if write_access is None:
            raise RuntimeError(
                "landlock package API not recognized. "
                'Run: python -c "import landlock; print(dir(landlock))" '
                "and adapt mapping for your version."
            )
        write_names = [
            "WRITE_FILE",
            "TRUNCATE",
            "MAKE_REG",
            "MAKE_DIR",
            "MAKE_SYM",
            "MAKE_FIFO",
            "MAKE_SOCK",
            "MAKE_CHAR",
            "MAKE_BLOCK",
            "REMOVE_FILE",
            "REMOVE_DIR",
            "REFER",
        ]
        try:
            mask = write_access(0)
            mask_is_enum = True
        except Exception:
            mask = 0
            mask_is_enum = False
        for name in write_names:
            if hasattr(write_access, name):
                value = getattr(write_access, name)
                mask |= value if mask_is_enum else int(value)
        if mask == 0:
            raise RuntimeError(
                "Could not build a write-access mask from landlock's exported flags."
            )
        try:
            rs = Ruleset(restrict_rules=mask)
        except TypeError as exc:
            raise RuntimeError(
                "landlock Ruleset does not support restrict_rules; cannot build write-only ruleset."
            ) from exc
        for root in cfg.writable_roots:
            try:
                rs.allow(root, rules=mask)
            except TypeError:
                rs.allow(root, access=mask)
        rs.apply()
        os.umask(cfg.umask)
        return None

    for root in cfg.writable_roots:
        rs.allow(root)
    rs.apply()
    os.umask(cfg.umask)
    return None


def _apply_sandbox(cfg: SandboxConfig) -> None:
    return _apply_sandbox_linux(cfg)


def _check_macos_sandbox_dependencies() -> list[str]:
    errors: list[str] = []
    if shutil.which("sandbox-exec") is None:
        errors.append("sandbox-exec not found")
    if not Path("/bin/sh").exists():
        errors.append("/bin/sh not found")
    return errors


def _get_sandbox_unavailable_reason(cfg: SandboxConfig) -> str | None:
    if _IS_LINUX:
        return None
    if _IS_MACOS:
        errors = _check_macos_sandbox_dependencies()
        if errors:
            return ", ".join(errors)
        return None
    return f"Sandbox unsupported on platform {platform.system()}"


def _ensure_sandbox_available(cfg: SandboxConfig) -> bool:
    reason = _get_sandbox_unavailable_reason(cfg)
    if reason is None:
        return True
    if cfg.fail_if_unavailable:
        raise RuntimeError(reason)
    _LOGGER.warning("Sandbox unavailable; running unsandboxed: %s", reason)
    return False


def _resolve_tmp_root(
    cfg: SandboxConfig,
    env: Mapping[str, str] | None,
) -> SandboxConfig:
    if cfg.tmp_root is not None:
        return cfg
    if not _IS_MACOS:
        return cfg
    env_map = env if env is not None else os.environ
    tmp_root = env_map.get("TMPDIR") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    return replace(cfg, tmp_root=tmp_root)


def _build_child_env(
    cfg: SandboxConfig,
    env: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if env is None and cfg.tmp_root is None:
        return None
    merged = dict(os.environ if env is None else env)
    if cfg.tmp_root is not None:
        merged["TMPDIR"] = cfg.tmp_root
    return merged


def _build_seatbelt_profile(cfg: SandboxConfig) -> str:
    """Build the macOS seatbelt profile for the resolved sandbox config."""
    rules = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow file-read*)",
    ]
    for path in cfg.deny_read:
        rules.extend(_seatbelt_path_rules("deny", "file-read*", path))
    for path in cfg.allow_read:
        rules.extend(_seatbelt_path_rules("allow", "file-read*", path))
    for path in cfg.allow_write:
        rules.extend(_seatbelt_path_rules("allow", "file-write*", path))
    for path in cfg.deny_write:
        rules.extend(_seatbelt_path_rules("deny", "file-write*", path))
    return "\n".join(rules)


def _macos_sandbox_wrap(args: Sequence[str], cfg: SandboxConfig) -> list[str]:
    profile = _build_seatbelt_profile(cfg)
    return ["sandbox-exec", "-p", profile, *list(args)]


def _linux_child_setup(cfg: SandboxConfig) -> Callable[[], None]:
    def _child_setup() -> None:
        _apply_sandbox_linux(cfg)
        if cfg.cwd:
            os.chdir(cfg.cwd)
        return None

    return _child_setup


def _macos_child_setup(cfg: SandboxConfig) -> Callable[[], None]:
    def _child_setup() -> None:
        _set_rlimits(
            cpu_seconds=cfg.cpu_seconds,
            as_bytes=cfg.as_bytes,
            fsize_bytes=cfg.fsize_bytes,
            nofile=cfg.nofile,
            nproc=cfg.nproc,
        )
        os.umask(cfg.umask)
        return None

    return _child_setup


_LINUX_LAUNCHER_PATH = Path(__file__).with_name("_sandbox_linux_launcher.py")


def _launcher_argv(
    argv: Sequence[str],
    *,
    cfg: SandboxConfig,
    env: Mapping[str, str] | None,
) -> list[str]:
    payload = {
        "writable_roots": list(cfg.writable_roots),
        "cwd": cfg.cwd,
        "umask": cfg.umask,
        "rlimits": {
            "cpu_seconds": cfg.cpu_seconds,
            "as_bytes": cfg.as_bytes,
            "fsize_bytes": cfg.fsize_bytes,
            "nofile": cfg.nofile,
            "nproc": cfg.nproc,
        },
        "argv": list(argv),
        "env": None if env is None else dict(env),
    }
    return [sys.executable, str(_LINUX_LAUNCHER_PATH), json.dumps(payload)]


def _prepare_config_and_env(
    cfg: SandboxConfig,
    env: Mapping[str, str] | None,
) -> tuple[SandboxConfig, dict[str, str] | None]:
    resolved_cfg = _resolve_tmp_root(cfg, env).normalized()
    child_env = _build_child_env(resolved_cfg, env)
    return resolved_cfg, child_env


def sandbox_popen(
    args: Sequence[str],
    *,
    cfg: SandboxConfig,
    stdin=None,
    stdout=None,
    stderr=None,
    env: Mapping[str, str] | None = None,
    text: bool = False,
) -> subprocess.Popen:
    """Spawn a subprocess using the strongest available sandbox for the platform."""
    resolved_cfg, child_env = _prepare_config_and_env(cfg, env)
    if _IS_LINUX:
        return subprocess.Popen(
            list(args),
            preexec_fn=_linux_child_setup(resolved_cfg),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=child_env,
            text=text,
            close_fds=True,
        )
    if _IS_MACOS and _ensure_sandbox_available(resolved_cfg):
        return subprocess.Popen(
            _macos_sandbox_wrap(args, resolved_cfg),
            preexec_fn=_macos_child_setup(resolved_cfg),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=child_env,
            text=text,
            close_fds=True,
            cwd=resolved_cfg.cwd,
        )
    return subprocess.Popen(
        list(args),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        env=child_env,
        text=text,
        close_fds=True,
        cwd=resolved_cfg.cwd,
    )


async def sandbox_exec_async(
    *argv: str,
    cfg: SandboxConfig,
    stdin=asyncio.subprocess.DEVNULL,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    env: Mapping[str, str] | None = None,
) -> asyncio.subprocess.Process:
    """Spawn an argv-style command using the strongest available sandbox."""
    resolved_cfg, child_env = _prepare_config_and_env(cfg, env)
    if _IS_LINUX:
        launcher = _launcher_argv(argv, cfg=resolved_cfg, env=child_env)
        return await asyncio.create_subprocess_exec(
            *launcher,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
    if _IS_MACOS and _ensure_sandbox_available(resolved_cfg):
        return await asyncio.create_subprocess_exec(
            *_macos_sandbox_wrap(argv, resolved_cfg),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=resolved_cfg.cwd,
            env=child_env,
            preexec_fn=_macos_child_setup(resolved_cfg),
        )
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=resolved_cfg.cwd,
        env=child_env,
    )


async def sandbox_shell_async(
    command: str,
    *,
    cfg: SandboxConfig,
    stdin=asyncio.subprocess.DEVNULL,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    env: Mapping[str, str] | None = None,
) -> asyncio.subprocess.Process:
    """Spawn a shell command using the strongest available sandbox."""
    resolved_cfg, child_env = _prepare_config_and_env(cfg, env)
    shell_argv = ("/bin/sh", "-c", command)
    if _IS_LINUX:
        launcher = _launcher_argv(shell_argv, cfg=resolved_cfg, env=child_env)
        return await asyncio.create_subprocess_exec(
            *launcher,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
    if _IS_MACOS and _ensure_sandbox_available(resolved_cfg):
        return await asyncio.create_subprocess_exec(
            *_macos_sandbox_wrap(shell_argv, resolved_cfg),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=resolved_cfg.cwd,
            env=child_env,
            preexec_fn=_macos_child_setup(resolved_cfg),
        )
    return await asyncio.create_subprocess_exec(
        *shell_argv,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=resolved_cfg.cwd,
        env=child_env,
    )
