#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
SERVICE_NAME="${TPCH_MONETDB_SERVICE:-tpch-monetdb}"
INIT_SERVICE_NAME="${TPCH_MONETDB_INIT_SERVICE:-tpch-monetdb-init}"
CONTAINER_NAME="${TPCH_MONETDB_CONTAINER:-tpch-monetdb}"
HEALTH_RETRIES="${TPCH_MONETDB_HEALTH_RETRIES:-60}"
HEALTH_SLEEP_SECONDS="${TPCH_MONETDB_HEALTH_SLEEP_SECONDS:-2}"
WORKSPACE_DIR="${TPCH_MONETDB_WORKSPACE:-/workspace/tpch_monetdb_project}"

export TPCH_MONETDB_IMAGE="${TPCH_MONETDB_IMAGE:-tpch-monetdb:local}"
export TPCH_MONETDB_CONTAINER="${TPCH_MONETDB_CONTAINER:-tpch-monetdb}"
export TPCH_MONETDB_PORT="${TPCH_MONETDB_PORT:-50000}"
export TPCH_MONETDB_DATABASE="${TPCH_MONETDB_DATABASE:-tpch_smoke}"
export TPCH_MONETDB_USER="${TPCH_MONETDB_USER:-monetdb}"
export TPCH_MONETDB_PASSWORD="${TPCH_MONETDB_PASSWORD:-monetdb}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export USE_ALIYUN_DNF_MIRROR="${USE_ALIYUN_DNF_MIRROR:-true}"
export TPCH_MONETDB_WORKSPACE="${TPCH_MONETDB_WORKSPACE:-/workspace/tpch_monetdb_project}"
export TPCH_MONETDB_WORKSPACE_HOST="${TPCH_MONETDB_WORKSPACE_HOST:-${REPO_ROOT}}"
export TPCH_MONETDB_HARDWARE_COUNTER_BACKEND="${TPCH_MONETDB_HARDWARE_COUNTER_BACKEND:-linux_perf_native}"
export TPCH_MONETDB_PERF_BINARY="${TPCH_MONETDB_PERF_BINARY:-/usr/local/bin/tpch-perf}"
export TPCH_MONETDB_PERF_EVENT_PARANOID="${TPCH_MONETDB_PERF_EVENT_PARANOID:-1}"

# Print usage for the deployment helper.
usage() {
  cat <<'EOF'
Usage:
  docker/tpch-monetdb/deploy.sh [command]

Commands:
  deploy       Build image, start service, import tiny TPC-H fixtures, run smoke check (default)
  build        Build the Compose image
  start        Start the long-lived TPC-H MonetDB service and wait for health
  init         Import bundled tiny TPC-H fixtures into the database
  smoke        Run an in-container SQL smoke check
  test [args]  Run pytest inside the container workspace as root
  shell        Open a bash shell inside the container workspace as root
  root-shell   Alias for shell
  monetdb-shell
              Open a bash shell inside the container workspace as monetdb
  workspace    Show the mounted workspace paths inside the container
  grant-perf   Set/confirm perf_event_paranoid from the privileged container
  pmu-smoke    Run an in-container perf stat smoke check
  ps           Show Compose service status
  logs         Tail MonetDB service logs
  stop         Stop the long-lived service
  down         Stop and remove Compose containers, keep dbfarm volume
  reset        Stop and remove Compose containers and dbfarm volume
  help         Show this help

Useful environment overrides:
  TPCH_MONETDB_PORT=50000
  TPCH_MONETDB_DATABASE=tpch_smoke
  TPCH_MONETDB_USER=monetdb
  TPCH_MONETDB_PASSWORD=monetdb
  TPCH_MONETDB_IMAGE=tpch-monetdb:local
  TPCH_MONETDB_WORKSPACE=/workspace/tpch_monetdb_project
  TPCH_MONETDB_WORKSPACE_HOST=/absolute/path/to/repo
  TPCH_MONETDB_PERF_BINARY=/usr/local/bin/tpch-perf
  TPCH_MONETDB_PERF_EVENT_PARANOID=1
  USE_ALIYUN_DNF_MIRROR=true
  PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
EOF
}

# Run docker compose with the project-local compose file.
compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

# Fail early when Docker or the Compose file is unavailable.
preflight() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed or not on PATH." >&2
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: docker compose plugin is unavailable." >&2
    exit 1
  fi
  if [[ ! -f "${COMPOSE_FILE}" ]]; then
    echo "ERROR: Compose file not found: ${COMPOSE_FILE}" >&2
    exit 1
  fi
}

# Build the TPC-H MonetDB image.
build_image() {
  compose build "${SERVICE_NAME}"
}

# Wait until the long-lived service reports healthy.
wait_for_health() {
  local attempt=1
  local status=""
  while (( attempt <= HEALTH_RETRIES )); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
    if [[ "${status}" == "healthy" ]]; then
      echo "Service ${CONTAINER_NAME} is healthy."
      return 0
    fi
    echo "Waiting for ${CONTAINER_NAME} health (${attempt}/${HEALTH_RETRIES}): ${status:-not-created}"
    sleep "${HEALTH_SLEEP_SECONDS}"
    attempt=$((attempt + 1))
  done
  echo "ERROR: ${CONTAINER_NAME} did not become healthy." >&2
  compose ps
  compose logs --tail=120 "${SERVICE_NAME}" || true
  exit 1
}

# Start the long-lived MonetDB service.
start_service() {
  compose up -d "${SERVICE_NAME}"
  wait_for_health
}

# Import the bundled tiny TPC-H fixture through the init profile.
init_fixture() {
  compose --profile init run --rm "${INIT_SERVICE_NAME}"
}

# Run a smoke query inside the service container without requiring host pymonetdb.
smoke_check() {
  compose exec -T "${SERVICE_NAME}" python - <<'PY'
import os
import pymonetdb

conn = pymonetdb.connect(
    hostname="127.0.0.1",
    port=int(os.environ.get("TPCH_MONETDB_PORT", "50000")),
    database=os.environ.get("TPCH_MONETDB_DATABASE", "tpch_smoke"),
    username=os.environ.get("TPCH_MONETDB_USER", "monetdb"),
    password=os.environ.get("TPCH_MONETDB_PASSWORD", "monetdb"),
    autocommit=True,
)
try:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM lineitem")
    row_count = cur.fetchone()[0]
    print(f"smoke_ok lineitem_rows={row_count}")
finally:
    conn.close()
PY
}

# Run pytest from the mounted repository inside the service container.
run_tests() {
  if (( $# == 0 )); then
    compose exec -T -u root "${SERVICE_NAME}" python -m pytest -p no:cacheprovider tpch_monetdb/tests -q
    return 0
  fi
  compose exec -T -u root "${SERVICE_NAME}" python -m pytest -p no:cacheprovider "$@"
}

# Open an interactive shell in the mounted workspace as root.
open_shell() {
  compose exec -u root --workdir "${WORKSPACE_DIR}" "${SERVICE_NAME}" bash
}

# Open an interactive shell in the mounted workspace as root for build/test writes.
open_root_shell() {
  compose exec -u root --workdir "${WORKSPACE_DIR}" "${SERVICE_NAME}" bash
}

# Open an interactive shell in the mounted workspace as the MonetDB service user.
open_monetdb_shell() {
  compose exec --workdir "${WORKSPACE_DIR}" "${SERVICE_NAME}" bash
}

# Print the mounted workspace location and key project paths inside the container.
show_workspace() {
  compose exec -T "${SERVICE_NAME}" bash -lc '
    set -e
    echo "workspace=${TPCH_MONETDB_WORKSPACE}"
    pwd
    ls -la "${TPCH_MONETDB_WORKSPACE}"
    test -d "${TPCH_MONETDB_WORKSPACE}/tpch_monetdb/misc/tpch"
    ls -la "${TPCH_MONETDB_WORKSPACE}/tpch_monetdb/misc/tpch" | sed -n "1,40p"
  '
}

# Grant perf_event access for Linux PMU collection from the privileged container.
grant_perf_permissions() {
  compose exec -T -u root \
    -e TPCH_MONETDB_PERF_EVENT_PARANOID="${TPCH_MONETDB_PERF_EVENT_PARANOID}" \
    "${SERVICE_NAME}" bash -lc '
    set -euo pipefail
    desired="${TPCH_MONETDB_PERF_EVENT_PARANOID:-1}"
    current="$(cat /proc/sys/kernel/perf_event_paranoid)"
    echo "perf_event_paranoid_before=${current}"
    if [ -w /proc/sys/kernel/perf_event_paranoid ]; then
      echo "${desired}" > /proc/sys/kernel/perf_event_paranoid
    fi
    echo "perf_event_paranoid_after=$(cat /proc/sys/kernel/perf_event_paranoid)"
  '
}

# Run a perf stat smoke test inside the service container.
pmu_smoke() {
  compose exec -T \
    -e TPCH_MONETDB_PERF_BINARY="${TPCH_MONETDB_PERF_BINARY}" \
    "${SERVICE_NAME}" bash -lc '
    set -euo pipefail
    perf_bin="${TPCH_MONETDB_PERF_BINARY:-/usr/local/bin/tpch-perf}"
    test -x "${perf_bin}"
    echo "kernel=$(uname -r)"
    echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
    "${perf_bin}" stat -x, \
      -e cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses \
      -- dd if=/dev/zero of=/dev/null bs=1M count=16
  '
}

# Execute the full deployment workflow.
deploy() {
  preflight
  build_image
  start_service
  init_fixture
  smoke_check
  compose ps
}

command="${1:-deploy}"
if (( $# > 0 )); then
  shift
fi
case "${command}" in
  deploy)
    deploy
    ;;
  build)
    preflight
    build_image
    ;;
  start)
    preflight
    start_service
    ;;
  init)
    preflight
    init_fixture
    ;;
  smoke)
    preflight
    smoke_check
    ;;
  test)
    preflight
    build_image
    start_service
    run_tests "$@"
    ;;
  shell)
    preflight
    start_service
    open_shell
    ;;
  root-shell)
    preflight
    start_service
    open_root_shell
    ;;
  monetdb-shell)
    preflight
    start_service
    open_monetdb_shell
    ;;
  workspace)
    preflight
    start_service
    show_workspace
    ;;
  grant-perf)
    preflight
    start_service
    grant_perf_permissions
    ;;
  pmu-smoke)
    preflight
    start_service
    pmu_smoke
    ;;
  ps)
    preflight
    compose ps
    ;;
  logs)
    preflight
    compose logs -f "${SERVICE_NAME}"
    ;;
  stop)
    preflight
    compose stop "${SERVICE_NAME}"
    ;;
  down)
    preflight
    compose down
    ;;
  reset)
    preflight
    compose down -v
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "ERROR: unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
