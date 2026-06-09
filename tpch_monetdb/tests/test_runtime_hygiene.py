import pytest
from pathlib import Path
from tpch_monetdb.tools.tpch.runtime_hygiene import (
    RuntimeHealthReport,
    RuntimeHealthThresholds,
    cleanup_reload_dir,
    classify_infra_failure,
    inspect_runtime_health,
)
from tpch_monetdb.tools.tpch.pool import FastTestPool


class TestRuntimeHygiene:
    def test_inspect_empty_reload_healthy(self, tmp_path: Path):
        reload_dir = tmp_path / "build" / ".reload"
        reload_dir.mkdir(parents=True)
        report = inspect_runtime_health(tmp_path)
        assert report.healthy
        assert report.reload_files == 0
        assert report.reload_bytes == 0

    def test_reload_file_limit_unhealthy(self, tmp_path: Path):
        reload_dir = tmp_path / "build" / ".reload"
        reload_dir.mkdir(parents=True)
        for i in range(200):
            (reload_dir / f"test_{i}.so").write_text("x")
        thresholds = RuntimeHealthThresholds(reload_max_files=128, reload_max_bytes=10 * 1024 * 1024 * 1024)
        report = inspect_runtime_health(tmp_path, thresholds)
        assert not report.healthy
        assert report.reason_code == "RELOAD_FILE_LIMIT"

    def test_reload_byte_limit_unhealthy(self, tmp_path: Path):
        reload_dir = tmp_path / "build" / ".reload"
        reload_dir.mkdir(parents=True)
        (reload_dir / "big.so").write_text("x" * 1024 * 1024)  # 1MB
        thresholds = RuntimeHealthThresholds(reload_max_files=10000, reload_max_bytes=1024)
        report = inspect_runtime_health(tmp_path, thresholds)
        assert not report.healthy
        assert report.reason_code == "RELOAD_BYTE_LIMIT"

    def test_cleanup_reload_dir(self, tmp_path: Path):
        reload_dir = tmp_path / "build" / ".reload"
        reload_dir.mkdir(parents=True)
        (reload_dir / "old.so").write_text("old")
        report = cleanup_reload_dir(tmp_path)
        assert report.healthy
        assert report.reload_files == 0

    def test_cleanup_creates_dir(self, tmp_path: Path):
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        report = cleanup_reload_dir(tmp_path)
        assert report.healthy
        assert (tmp_path / "build" / ".reload").is_dir()


class TestClassifyInfraFailure:
    def test_fork_enomem(self):
        code = classify_infra_failure("fork: Cannot allocate memory")
        assert code == "FORK_ENOMEM"

    def test_infra_blocked_fork(self):
        code = classify_infra_failure("[ERROR:INFRA_BLOCKED] fork failed: Cannot allocate memory")
        assert code == "FORK_ENOMEM"

    def test_result_csv_missing(self):
        code = classify_infra_failure("Expected output file missing")
        assert code == "RESULT_CSV_MISSING"

    def test_signal_11_preempts_result_csv_missing(self) -> None:
        text = "resp=exit_code: 0 signal: 11\nExpected output file missing: result1.csv"
        code = classify_infra_failure(text)
        assert code == "RUNNER_SEGFAULT"
        return None

    def test_metrics_result_missing_does_not_hide_signal_11(self) -> None:
        text = "resp=exit_code: 0 signal: 11\nExpected output file missing: result1.csv"
        metrics = {"validation/failure_code": "RESULT_CSV_MISSING"}
        code = classify_infra_failure(text, metrics)
        assert code == "RUNNER_SEGFAULT"
        return None

    def test_child_terminates_with_clean_exit_and_success_markers_is_not_infra_failure(self) -> None:
        text = (
            "2 | Execution ms: 0.096\n"
            "2 | Query ms: 0.096\n"
            "query done\n"
            "./build/libquery.so child terminates\n"
            "./build/libbuilder.so child terminates\n"
            "exit_code: 0 signal: 0\n"
        )
        code = classify_infra_failure(text)
        assert code is None
        return None

    def test_child_terminates_with_nonzero_exit_is_infra_failure(self) -> None:
        text = (
            "./build/libquery.so child terminates\n"
            "exit_code: 1 signal: 0\n"
        )
        code = classify_infra_failure(text)
        assert code == "RUNNER_SEGFAULT"
        return None

    def test_child_terminates_with_clean_exit_and_no_other_failures_is_not_infra_failure(self) -> None:
        text = (
            "./build/libquery.so child terminates\n"
            "exit_code: 0 signal: 0\n"
        )
        code = classify_infra_failure(text)
        assert code is None
        return None

    def test_process_exited_with_nonzero_code_is_infra_failure(self) -> None:
        code = classify_infra_failure("process exited with code -11")
        assert code == "RUNNER_SEGFAULT"
        return None

    def test_process_exited_with_zero_code_is_not_infra_failure(self) -> None:
        code = classify_infra_failure("process exited with code 0")
        assert code is None
        return None

    def test_runner_broken_pipe(self):
        code = classify_infra_failure("[ERROR:RUNNER_BROKEN_PIPE] Runner transport failed")
        assert code == "RUNNER_BROKEN_PIPE"

    def test_runner_timeout(self):
        code = classify_infra_failure("Terminated after 30 seconds due to timeout")
        assert code == "RUNNER_TIMEOUT"

    def test_metrics_failure_code_passthrough(self):
        metrics = {"validation/failure_code": "RUNNER_BROKEN_PIPE"}
        code = classify_infra_failure("some text", metrics)
        assert code == "RUNNER_BROKEN_PIPE"

    def test_no_match_returns_none(self):
        code = classify_infra_failure("normal validation output passed")
        assert code is None
