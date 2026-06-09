from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _ensure_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return None


def main() -> None:
    _ensure_repo_root()
    from tpch_monetdb.tools.sandbox import SandboxConfig, _apply_sandbox_linux

    payload = json.loads(sys.argv[1])
    rlimits = payload.get("rlimits", {})
    cfg = SandboxConfig(
        writable_roots=tuple(payload["writable_roots"]),
        cwd=payload.get("cwd"),
        cpu_seconds=rlimits.get("cpu_seconds"),
        as_bytes=rlimits.get("as_bytes"),
        fsize_bytes=rlimits.get("fsize_bytes"),
        nofile=rlimits.get("nofile"),
        nproc=rlimits.get("nproc"),
        umask=payload.get("umask", 0o077),
    )
    _apply_sandbox_linux(cfg)
    if cfg.cwd:
        os.chdir(cfg.cwd)
    argv = payload["argv"]
    os.execvpe(argv[0], argv, payload.get("env") or os.environ)
    return None


if __name__ == "__main__":
    main()
