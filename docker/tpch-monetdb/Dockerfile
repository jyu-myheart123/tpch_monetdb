FROM monetdb/monetdb:Dec2025-SP2

USER root

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com
ARG USE_ALIYUN_DNF_MIRROR=true
ARG WORKSPACE_DIR=/workspace/tpch_monetdb_project

RUN set -eux; \
    if [ "${USE_ALIYUN_DNF_MIRROR}" = "true" ]; then \
      for repo in /etc/yum.repos.d/*.repo; do \
        [ -f "${repo}" ] || continue; \
        sed -i \
          -e 's|^mirrorlist=|#mirrorlist=|g' \
          -e 's|^#baseurl=http://mirror.centos.org|baseurl=https://mirrors.aliyun.com|g' \
          -e 's|^#baseurl=https://mirror.centos.org|baseurl=https://mirrors.aliyun.com|g' \
          -e 's|^#baseurl=http://download.example/pub|baseurl=https://mirrors.aliyun.com|g' \
          "${repo}" || true; \
        done; \
    fi; \
    dnf install -y python3-pip gcc-c++ binutils git make pkgconf-pkg-config perf procps-ng; \
    perf_bin="$(command -v perf || true)"; \
    test -n "${perf_bin}"; \
    ln -sf "${perf_bin}" /usr/local/bin/tpch-perf; \
    python3 -m venv /opt/tpch-monetdb/venv; \
    /opt/tpch-monetdb/venv/bin/pip install --no-cache-dir --upgrade \
      -i "${PIP_INDEX_URL}" \
      --trusted-host "${PIP_TRUSTED_HOST}" \
      pip setuptools wheel; \
    dnf -y clean all; \
    rm -rf /tmp/* /var/tmp/* /var/cache/dnf

COPY . ${WORKSPACE_DIR}/

RUN set -eux; \
    /opt/tpch-monetdb/venv/bin/pip install --no-cache-dir \
      -i "${PIP_INDEX_URL}" \
      --trusted-host "${PIP_TRUSTED_HOST}" \
      -e "${WORKSPACE_DIR}" \
      pymonetdb==1.9.0; \
    mkdir -p /var/monetdb5/tpch-copy "${WORKSPACE_DIR}"; \
    chown -R monetdb:monetdb /opt/tpch-monetdb /var/monetdb5/tpch-copy "${WORKSPACE_DIR}"

USER monetdb

ENV PATH="/opt/tpch-monetdb/venv/bin:${PATH}" \
    PYTHONPATH=/workspace/tpch_monetdb_project \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MDB_CREATE_DBS=tpch_smoke \
    MDB_FARM_DIR=/var/monetdb5/dbfarm \
    TPCH_MONETDB_WORKSPACE=/workspace/tpch_monetdb_project \
    TPCH_MONETDB_HARDWARE_COUNTER_BACKEND=linux_perf_native \
    TPCH_MONETDB_PERF_BINARY=/usr/local/bin/tpch-perf \
    TPCH_MONETDB_DATABASE=tpch_smoke \
    TPCH_MONETDB_USER=monetdb \
    TPCH_MONETDB_HOST=127.0.0.1 \
    TPCH_MONETDB_PORT=50000 \
    TPCH_MONETDB_FIXTURE_DIR=/data/tpch/sf1 \
    TMPDIR=/var/monetdb5/tpch-copy

WORKDIR /workspace/tpch_monetdb_project
