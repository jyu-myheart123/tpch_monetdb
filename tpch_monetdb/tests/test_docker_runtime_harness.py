from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = REPO_ROOT / "docker" / "tpch-monetdb"
EXPECTED_FIXTURE_FILES = {
    "customer.tbl",
    "lineitem.tbl",
    "nation.tbl",
    "orders.tbl",
    "part.tbl",
    "partsupp.tbl",
    "region.tbl",
    "supplier.tbl",
}
EXPECTED_FIXTURE_FIELD_COUNTS = {
    "customer.tbl": 8,
    "lineitem.tbl": 16,
    "nation.tbl": 4,
    "orders.tbl": 9,
    "part.tbl": 9,
    "partsupp.tbl": 5,
    "region.tbl": 3,
    "supplier.tbl": 7,
}


def read_harness_file(relative_path: str) -> str:
    """Read a Docker harness file as UTF-8 text for static contract tests."""
    return (HARNESS_ROOT / relative_path).read_text(encoding="utf-8")


def test_docker_compose_files_are_present() -> None:
    """Verify the Docker path is compose-first and exposes a deployment helper."""
    required_paths = [
        "Dockerfile",
        "docker-compose.yml",
        "deploy.sh",
    ]
    missing = [path for path in required_paths if not (HARNESS_ROOT / path).is_file()]
    assert missing == []
    assert (HARNESS_ROOT / "deploy.sh").stat().st_mode & 0o111
    assert (REPO_ROOT / ".dockerignore").is_file()
    assert not (HARNESS_ROOT / "scripts" / "entrypoint").exists()
    assert not (HARNESS_ROOT / "scripts" / "smoke.py").exists()
    return None


def test_dockerfile_uses_pinned_monetdb_base_and_aliyun_pip_source() -> None:
    """Verify the image is pinned and contains the complete project workspace."""
    dockerfile = read_harness_file("Dockerfile")
    assert "FROM monetdb/monetdb:Dec2025-SP2" in dockerfile
    assert "FROM --platform=" not in dockerfile
    assert "FROM monetdb/monetdb:latest" not in dockerfile
    assert "https://mirrors.aliyun.com/pypi/simple" in dockerfile
    assert "USE_ALIYUN_DNF_MIRROR=true" in dockerfile
    assert "gcc-c++" in dockerfile
    assert "binutils" in dockerfile
    assert "git" in dockerfile
    assert "make" in dockerfile
    assert "pkgconf-pkg-config" in dockerfile
    assert "perf procps-ng" in dockerfile
    assert "ln -sf \"${perf_bin}\" /usr/local/bin/tpch-perf" in dockerfile
    assert "pymonetdb==1.9.0" in dockerfile
    assert "COPY . ${WORKSPACE_DIR}/" in dockerfile
    assert "-e \"${WORKSPACE_DIR}\"" in dockerfile
    assert "mkdir -p /var/monetdb5/tpch-copy" in dockerfile
    assert "TMPDIR=/var/monetdb5/tpch-copy" in dockerfile
    assert "PYTHONPATH=/workspace/tpch_monetdb_project" in dockerfile
    assert "TPCH_MONETDB_WORKSPACE=/workspace/tpch_monetdb_project" in dockerfile
    assert "TPCH_MONETDB_HARDWARE_COUNTER_BACKEND=linux_perf_native" in dockerfile
    assert "TPCH_MONETDB_PERF_BINARY=/usr/local/bin/tpch-perf" in dockerfile
    assert "WORKDIR /workspace/tpch_monetdb_project" in dockerfile
    assert "COPY tpch_monetdb/dataset/" not in dockerfile
    assert "COPY tpch_monetdb/oracle/" not in dockerfile
    assert "ENTRYPOINT" not in dockerfile
    assert "CMD [" not in dockerfile
    assert "scripts/smoke.py" not in dockerfile
    assert "TPCH_MONETDB_PASSWORD=" not in dockerfile
    assert "MDB_DB_ADMIN_PASS=" not in dockerfile
    return None


def test_compose_declares_monetdb_service_and_init_profile() -> None:
    """Verify Compose owns service startup and fixture import."""
    compose = read_harness_file("docker-compose.yml")
    assert "services:" in compose
    assert "tpch-monetdb:" in compose
    assert "tpch-monetdb-init:" in compose
    assert "\n  monetdb:" not in compose
    assert "\n  tpch-init:" not in compose
    assert "profiles:" in compose
    assert "- init" in compose
    assert "dockerfile: docker/tpch-monetdb/Dockerfile" in compose
    assert "${TPCH_MONETDB_IMAGE:-tpch-monetdb:local}" in compose
    assert "privileged: true" in compose
    assert "cap_add:" in compose
    assert "- SYS_ADMIN" in compose
    assert "- SYS_PTRACE" in compose
    assert "- PERFMON" not in compose
    assert "security_opt:" in compose
    assert "- seccomp=unconfined" in compose
    assert "ulimits:" in compose
    assert "memlock:" in compose
    assert "${TPCH_MONETDB_PORT:-50000}:50000" in compose
    assert "MDB_CREATE_DBS: ${TPCH_MONETDB_DATABASE:-tpch_smoke}" in compose
    assert "MDB_DB_ADMIN_PASS: ${TPCH_MONETDB_PASSWORD:-monetdb}" in compose
    assert "./fixtures/tiny-tpch:/data/tpch/sf1:ro" in compose
    assert "${TPCH_MONETDB_WORKSPACE_HOST:-../..}:${TPCH_MONETDB_WORKSPACE:-/workspace/tpch_monetdb_project}" in compose
    assert "TPCH_MONETDB_WORKSPACE: ${TPCH_MONETDB_WORKSPACE:-/workspace/tpch_monetdb_project}" in compose
    assert "TPCH_MONETDB_HARDWARE_COUNTER_BACKEND: ${TPCH_MONETDB_HARDWARE_COUNTER_BACKEND:-linux_perf_native}" in compose
    assert "TPCH_MONETDB_PERF_BINARY: ${TPCH_MONETDB_PERF_BINARY:-/usr/local/bin/tpch-perf}" in compose
    assert "TPCH_MONETDB_PERF_EVENT_PARANOID: ${TPCH_MONETDB_PERF_EVENT_PARANOID:-1}" in compose
    assert "PYTHONPATH: ${TPCH_MONETDB_WORKSPACE:-/workspace/tpch_monetdb_project}" in compose
    assert "tpch-monetdb-copy:/var/monetdb5/tpch-copy" in compose
    assert "TMPDIR: /var/monetdb5/tpch-copy" in compose
    assert "TPCH_MONETDB_HOST: tpch-monetdb" in compose
    assert "condition: service_healthy" in compose
    assert "prepare_tpch_database" in compose
    assert "DROP TABLE IF EXISTS" in compose
    assert "scripts/entrypoint" not in compose
    assert "smoke.py" not in compose
    return None


def test_deploy_script_wraps_compose_workflow_without_custom_entrypoint() -> None:
    """Verify deploy.sh orchestrates Compose without replacing service startup."""
    script = read_harness_file("deploy.sh")
    assert "docker compose -f \"${COMPOSE_FILE}\"" in script
    assert "compose build \"${SERVICE_NAME}\"" in script
    assert "compose up -d \"${SERVICE_NAME}\"" in script
    assert "compose --profile init run --rm \"${INIT_SERVICE_NAME}\"" in script
    assert "wait_for_health" in script
    assert "SELECT COUNT(*) FROM lineitem" in script
    assert "python -m pytest -p no:cacheprovider" in script
    assert "compose exec -T -u root \"${SERVICE_NAME}\"" in script
    assert "compose exec -u root --workdir \"${WORKSPACE_DIR}\" \"${SERVICE_NAME}\" bash" in script
    assert "root-shell" in script
    assert "monetdb-shell" in script
    assert "workspace" in script
    assert "grant-perf" in script
    assert "pmu-smoke" in script
    assert "TPCH_MONETDB_PERF_BINARY" in script
    assert "TPCH_MONETDB_PERF_EVENT_PARANOID" in script
    assert "-e TPCH_MONETDB_PERF_EVENT_PARANOID=\"${TPCH_MONETDB_PERF_EVENT_PARANOID}\"" in script
    assert "-e TPCH_MONETDB_PERF_BINARY=\"${TPCH_MONETDB_PERF_BINARY}\"" in script
    assert "perf_event_paranoid_before" in script
    assert "cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses" in script
    assert "tpch_monetdb/misc/tpch" in script
    assert "scripts/entrypoint" not in script
    assert "smoke.py" not in script
    return None


def test_dockerignore_excludes_local_build_artifacts() -> None:
    """Verify Docker build context excludes local caches while keeping source visible."""
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".git" in dockerignore
    assert ".venv/" in dockerignore
    assert ".pytest_cache/" in dockerignore
    assert "wandb/" in dockerignore
    assert "tpch_monetdb_artifacts*/" in dockerignore
    assert "tpch_monetdb/" not in dockerignore
    assert "docker/" not in dockerignore
    return None


def test_docker_harness_contains_no_binary_files() -> None:
    """Verify the compose directory is plain text plus tiny TPC-H fixtures."""
    for path in HARNESS_ROOT.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        assert b"\x00" not in data, path.relative_to(HARNESS_ROOT).as_posix()
        data.decode("utf-8")
    return None


def test_tiny_tpch_fixture_contains_all_required_table_files() -> None:
    """Verify that the bundled tiny fixture exposes the eight TPC-H table files."""
    fixture_dir = HARNESS_ROOT / "fixtures" / "tiny-tpch"
    fixture_files = {path.name for path in fixture_dir.glob("*.tbl")}
    assert fixture_files == EXPECTED_FIXTURE_FILES
    for path in fixture_dir.glob("*.tbl"):
        text = path.read_text(encoding="utf-8")
        assert text.endswith("|\n")
        assert "|" in text
        fields = text.rstrip("\n").split("|")[:-1]
        assert len(fields) == EXPECTED_FIXTURE_FIELD_COUNTS[path.name]
    return None
