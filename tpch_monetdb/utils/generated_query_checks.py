from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from tpch_monetdb.utils.pipeline_evidence import FORBIDDEN_FINAL_PATH_TOKENS
from tpch_monetdb.utils.query_codegen_hints import QUERY_CODEGEN_HINTS
from tpch_monetdb.utils.query_units import build_query_unit_lookup


@dataclass(frozen=True)
class GeneratedCheckViolation:
    code: str
    severity: str
    message: str
    file_path: str


_RESULT_FILE_RE = re.compile(r"result\d+\.csv")
_EXECUTE_ENTRYPOINT_RE = re.compile(r"\bexecute_q\d+\s*\(")
_USAGE_INT_CAST_RE = re.compile(
    r"(?:static_cast\s*<\s*(?:int64_t|long|int)\s*>\s*\(\s*[^)]*usage_|"
    r"\(\s*(?:int64_t|long|int)\s*\)\s*[^;\n]*usage_)"
)
_FORBIDDEN_FINAL_PLACEHOLDER_RE = re.compile(
    r"\b(?:stub|placeholder|unimplemented)\b",
    re.IGNORECASE,
)
_RAW_SOURCE_RECONSTRUCTION_RE = re.compile(
    r"(?:\b(?:std::)?i?fstream\b|\bstd::ifstream\b|\bstd::ofstream\b|"
    r"\bfopen\s*\(|\.ilp\b|cpu\.ilp|source\s+ilp|raw\s+source|"
    r"source[-_ ]path|base_data_dir|TPCH_MONETDB_QUERY_BATCH_FILE)",
    re.IGNORECASE,
)
_EMPTY_CSV_ASSIGNMENT_RE = re.compile(
    r"(?:\bcsv_output\s*=\s*\"\"\s*;|\.csv_output\s*=\s*\"\"\s*;)"
)
_CSV_OUTPUT_CLEAR_RE = re.compile(r"\.csv_output\.clear\s*\(")
_CSV_OUTPUT_POPULATION_RE = re.compile(
    r"(?:\bcsv_output\s*=\s*(?!\s*\"\"\s*;)|"
    r"\.csv_output\s*=\s*(?!\s*\"\"\s*;)|"
    r"\.csv_output\s*(?:\+=|\.append\s*\(|\.assign\s*\(|\.push_back\s*\())"
)
_CSV_OUTPUT_VALID_TRUE_RE = re.compile(
    r"(?:\bvalid|\.[A-Za-z_][A-Za-z0-9_]*\.valid|\.valid)\s*=\s*true\s*;"
)
_MANUAL_REGISTRY_FALLBACK_RE = re.compile(
    r"(?:\bdispatch_unimplemented_query\s*\(|Unimplemented query module)",
    re.IGNORECASE,
)
_SUPPORTED_GENERATED_CODE_CHECKS = frozenset(
    {
        "query_protocol",
        "query_antipatterns",
        "query_family_boundary",
        "final_path_integrity",
        "usage_double_output",
        "critical_vector_loop_shape",
    }
)


def run_generated_code_checks(
    *,
    workspace_root: Path,
    expected_query_id: str | None,
    checks: tuple[str, ...],
    active_unit_files: tuple[str, ...] = (),
) -> list[GeneratedCheckViolation]:
    violations: list[GeneratedCheckViolation] = []
    for check_name in checks:
        if check_name not in _SUPPORTED_GENERATED_CODE_CHECKS:
            raise ValueError(f"Unknown generated_code_check: {check_name}")
        if check_name == "query_protocol":
            if expected_query_id is None:
                continue
            violations.extend(
                _check_query_protocol(
                    workspace_root=workspace_root,
                    expected_query_id=expected_query_id,
                )
            )
        if check_name == "query_antipatterns":
            if expected_query_id is None:
                continue
            violations.extend(
                _check_query_antipatterns(
                    workspace_root=workspace_root,
                    expected_query_id=expected_query_id,
                )
            )
        if check_name == "query_family_boundary":
            violations.extend(
                _check_query_family_boundaries(
                    workspace_root=workspace_root,
                    active_unit_files=active_unit_files,
                )
            )
        if check_name == "final_path_integrity":
            if expected_query_id is None:
                continue
            violations.extend(
                _check_final_path_integrity(
                    workspace_root=workspace_root,
                    expected_query_id=expected_query_id,
                    active_unit_files=active_unit_files,
                )
            )
        if check_name == "usage_double_output":
            if expected_query_id is None:
                continue
            violations.extend(
                _check_usage_double_output(
                    workspace_root=workspace_root,
                    expected_query_id=expected_query_id,
                )
            )
        if check_name == "critical_vector_loop_shape":
            if expected_query_id is None:
                continue
            violations.extend(
                _check_critical_vector_loop_shape(
                    workspace_root=workspace_root,
                    expected_query_id=expected_query_id,
                )
            )
    return violations


def _check_query_protocol(
    *,
    workspace_root: Path,
    expected_query_id: str,
) -> list[GeneratedCheckViolation]:
    """Validate entrypoint and output-protocol rules for one generated query."""
    query_cpp = workspace_root / f"query_q{expected_query_id}.cpp"
    query_hpp = workspace_root / f"query_q{expected_query_id}.hpp"
    violations: list[GeneratedCheckViolation] = []
    expected_symbol = f"execute_q{expected_query_id}("
    if query_cpp.exists():
        cpp_text = query_cpp.read_text(encoding="utf-8")
        if expected_symbol not in cpp_text:
            violations.append(
                GeneratedCheckViolation(
                    code="MISSING_QUERY_ENTRYPOINT",
                    severity="error",
                    message=f"{query_cpp.name} is missing {expected_symbol}",
                    file_path=query_cpp.as_posix(),
                )
            )
    else:
        violations.append(
            GeneratedCheckViolation(
                code="MISSING_QUERY_SOURCE",
                severity="error",
                message=f"{query_cpp.name} is missing",
                file_path=query_cpp.as_posix(),
            )
        )
    if query_hpp.exists():
        hpp_text = query_hpp.read_text(encoding="utf-8")
        if expected_symbol not in hpp_text:
            violations.append(
                GeneratedCheckViolation(
                    code="MISSING_QUERY_DECLARATION",
                    severity="error",
                    message=f"{query_hpp.name} is missing {expected_symbol}",
                    file_path=query_hpp.as_posix(),
                )
            )
    else:
        violations.append(
            GeneratedCheckViolation(
                code="MISSING_QUERY_HEADER",
                severity="error",
                message=f"{query_hpp.name} is missing",
                file_path=query_hpp.as_posix(),
            )
        )
    for file_path in _iter_protocol_scoped_files(
        workspace_root=workspace_root,
        expected_query_id=expected_query_id,
    ):
        text = file_path.read_text(encoding="utf-8")
        result_match = _RESULT_FILE_RE.search(text)
        if result_match is not None:
            violations.append(
                GeneratedCheckViolation(
                    code="FORBIDDEN_RESULT_FILE_LITERAL",
                    severity="error",
                    message=(
                        f"{file_path.name} hardcodes {result_match.group(0)}; "
                        "query modules must use the shared runtime-owned output protocol"
                    ),
                    file_path=file_path.as_posix(),
                )
            )
        if _has_unguarded_csv_materialization(text):
            violations.append(
                GeneratedCheckViolation(
                    code="UNGUARDED_CSV_OUTPUT_MATERIALIZATION",
                    severity="error",
                    message=(
                        f"{file_path.name} populates csv_output or marks it valid without "
                        "checking should_materialize_query_output(); no-output timing must "
                        "not build CSV payloads."
                    ),
                    file_path=file_path.as_posix(),
                )
            )
    return violations


def _iter_protocol_scoped_files(
    *,
    workspace_root: Path,
    expected_query_id: str,
) -> list[Path]:
    candidates: list[Path] = []
    for relative_name in (
        f"query_q{expected_query_id}.cpp",
        f"query_q{expected_query_id}.hpp",
    ):
        target = workspace_root / relative_name
        if target.exists():
            candidates.append(target)
    candidates.extend(sorted(workspace_root.glob("query_shared_*.cpp")))
    candidates.extend(sorted(workspace_root.glob("query_shared_*.hpp")))
    return candidates


def _check_query_antipatterns(
    *,
    workspace_root: Path,
    expected_query_id: str,
) -> list[GeneratedCheckViolation]:
    """Check workload-specific anti-patterns for the active query."""
    hint = QUERY_CODEGEN_HINTS.get(str(expected_query_id))
    if hint is None:
        return []
    query_cpp = workspace_root / f"query_q{expected_query_id}.cpp"
    if not query_cpp.exists():
        return []
    source_text = query_cpp.read_text(encoding="utf-8")
    violations: list[GeneratedCheckViolation] = []
    for diagnostic in hint.diagnostics:
        violation = _match_antipattern(
            anti_pattern=diagnostic,
            query_id=expected_query_id,
            file_path=query_cpp,
            source_text=source_text,
            severity="diagnostic",
        )
        if violation is not None:
            violations.append(violation)
    return violations


def _check_final_path_integrity(
    *,
    workspace_root: Path,
    expected_query_id: str,
    active_unit_files: tuple[str, ...] = (),
) -> list[GeneratedCheckViolation]:
    """Reject final query code that routes through instrumentation or fallback code."""
    violations: list[GeneratedCheckViolation] = []
    for file_path in _iter_final_path_scoped_files(
        workspace_root=workspace_root,
        expected_query_id=expected_query_id,
        active_unit_files=active_unit_files,
    ):
        source_text = file_path.read_text(encoding="utf-8")
        for token in FORBIDDEN_FINAL_PATH_TOKENS:
            if token not in source_text:
                continue
            violations.append(
                GeneratedCheckViolation(
                    code="FORBIDDEN_INSTRUMENTED_FINAL_PATH",
                    severity="error",
                    message=(
                        f"{file_path.name} references instrumentation-only token "
                        f"`{token}`; final benchmark paths must call the "
                        "non-instrumented implementation path."
                    ),
                    file_path=file_path.as_posix(),
                )
            )
        violations.extend(
            _check_no_fallback_source(
                source_text=source_text,
                file_path=file_path,
            )
        )
    violations.extend(_check_manual_registry_fallback(workspace_root))
    return violations


def _iter_final_path_scoped_files(
    *,
    workspace_root: Path,
    expected_query_id: str,
    active_unit_files: tuple[str, ...],
) -> tuple[Path, ...]:
    """Return query-owned files whose code can affect a final query path."""
    candidates: list[Path] = []
    for relative_name in (
        f"query_q{expected_query_id}.cpp",
        f"query_q{expected_query_id}.hpp",
    ):
        candidates.append(workspace_root / relative_name)
    for relative_name in active_unit_files:
        if not (
            relative_name.startswith("query_q")
            or relative_name.startswith("query_family_")
            or relative_name.startswith("query_shared_")
        ):
            continue
        candidates.append(workspace_root / relative_name)
    candidates.extend(sorted(workspace_root.glob("query_shared_*.cpp")))
    candidates.extend(sorted(workspace_root.glob("query_shared_*.hpp")))
    existing_by_path = {
        path.resolve().as_posix(): path
        for path in candidates
        if path.exists()
    }
    return tuple(existing_by_path[key] for key in sorted(existing_by_path))


def _check_no_fallback_source(
    *,
    source_text: str,
    file_path: Path,
) -> list[GeneratedCheckViolation]:
    """Reject final-path fallback code that can make gates pass without evidence."""
    violations: list[GeneratedCheckViolation] = []
    if _FORBIDDEN_FINAL_PLACEHOLDER_RE.search(source_text):
        violations.append(
            GeneratedCheckViolation(
                code="FORBIDDEN_FINAL_PATH_PLACEHOLDER",
                severity="error",
                message=(
                    f"{file_path.name} contains stub/placeholder/unimplemented text; "
                    "final query paths must fail fast instead of carrying fallback code."
                ),
                file_path=file_path.as_posix(),
            )
        )
    if _RAW_SOURCE_RECONSTRUCTION_RE.search(source_text):
        violations.append(
            GeneratedCheckViolation(
                code="FORBIDDEN_RAW_SOURCE_RECONSTRUCTION",
                severity="error",
                message=(
                    f"{file_path.name} appears to open or rediscover raw TPC-H/source data; "
                    "query kernels must consume Engine data built by loader/builder."
                ),
                file_path=file_path.as_posix(),
            )
        )
    if _has_forbidden_empty_csv_output(source_text):
        violations.append(
            GeneratedCheckViolation(
                code="FORBIDDEN_EMPTY_CSV_OUTPUT",
                severity="error",
                message=(
                    f"{file_path.name} emits an empty CSV output path; "
                    "final query code must produce validated rows or fail."
                ),
                file_path=file_path.as_posix(),
            )
        )
    return violations


def _has_forbidden_empty_csv_output(source_text: str) -> bool:
    has_empty_marker = (
        _EMPTY_CSV_ASSIGNMENT_RE.search(source_text) is not None
        or _CSV_OUTPUT_CLEAR_RE.search(source_text) is not None
    )
    if not has_empty_marker:
        return False
    has_population = _CSV_OUTPUT_POPULATION_RE.search(source_text) is not None
    has_valid_true = _CSV_OUTPUT_VALID_TRUE_RE.search(source_text) is not None
    if has_population and has_valid_true:
        return False
    return True


def _has_unguarded_csv_materialization(source_text: str) -> bool:
    """Detect CSV payload writes that are not gated by the runtime output mode."""
    writes_csv_payload = (
        _CSV_OUTPUT_POPULATION_RE.search(source_text) is not None
        or _CSV_OUTPUT_VALID_TRUE_RE.search(source_text) is not None
    )
    if not writes_csv_payload:
        return False
    return "should_materialize_query_output" not in source_text


def _check_manual_registry_fallback(
    workspace_root: Path,
) -> list[GeneratedCheckViolation]:
    """Reject generated or hand-written registry fallback paths that mask missing queries."""
    violations: list[GeneratedCheckViolation] = []
    for relative_path in (
        Path("build/generated/query_registry_generated.cpp"),
        Path("generated/query_registry_generated.cpp"),
        Path("query_registry_generated.cpp"),
    ):
        registry_path = workspace_root / relative_path
        if not registry_path.exists():
            continue
        registry_text = registry_path.read_text(encoding="utf-8")
        if _MANUAL_REGISTRY_FALLBACK_RE.search(registry_text) is None:
            continue
        violations.append(
            GeneratedCheckViolation(
                code="FORBIDDEN_REGISTRY_FALLBACK",
                severity="error",
                message=(
                    f"{relative_path.as_posix()} contains a generated registry fallback; "
                    "missing query entrypoints must fail instead of routing through "
                    "dispatch_unimplemented_query."
                ),
                file_path=registry_path.as_posix(),
            )
        )
    return violations


def _check_usage_double_output(
    *,
    workspace_root: Path,
    expected_query_id: str,
) -> list[GeneratedCheckViolation]:
    """Reject usage_* metric output paths that narrow DOUBLE values to integers."""
    violations: list[GeneratedCheckViolation] = []
    for file_path in _iter_protocol_scoped_files(
        workspace_root=workspace_root,
        expected_query_id=expected_query_id,
    ):
        source_text = file_path.read_text(encoding="utf-8")
        if _USAGE_INT_CAST_RE.search(source_text) is None:
            continue
        violations.append(
            GeneratedCheckViolation(
                code="UNSAFE_USAGE_INT_CAST",
                severity="error",
                message=(
                    f"{file_path.name} casts usage_* DOUBLE metric values to an "
                    "integer output type; generated query code must preserve "
                    "lossless DOUBLE output semantics."
                ),
                file_path=file_path.as_posix(),
            )
        )
    return violations


def _check_critical_vector_loop_shape(
    *,
    workspace_root: Path,
    expected_query_id: str,
) -> list[GeneratedCheckViolation]:
    """Require critical vector queries to have a non-wrapper hot-loop owner file."""
    query_id = str(expected_query_id)
    violations: list[GeneratedCheckViolation] = []
    unit = build_query_unit_lookup([query_id]).get(query_id)
    if unit is None:
        return []
    hot_loop_files = tuple(
        relative_path for relative_path in unit.kernel_files
        if relative_path.endswith(".cpp")
    )
    missing_files = [
        relative_path
        for relative_path in hot_loop_files
        if not (workspace_root / relative_path).exists()
    ]
    if missing_files:
        violations.append(
            GeneratedCheckViolation(
                code="CRITICAL_VECTOR_HOT_LOOP_OWNER_MISSING",
                severity="diagnostic",
                message=(
                    "Critical vector query "
                    f"Q{query_id} must own its hot loop in: "
                    + ", ".join(missing_files)
                ),
                file_path=(workspace_root / f"query_q{query_id}.cpp").as_posix(),
            )
        )
        return violations
    for relative_path in hot_loop_files:
        source_path = workspace_root / relative_path
        source_text = source_path.read_text(encoding="utf-8")
        lowered = source_text.lower()
        if _looks_like_pending_vector_placeholder(lowered):
            violations.append(
                GeneratedCheckViolation(
                    code="CRITICAL_VECTOR_PLACEHOLDER",
                    severity="diagnostic",
                    message=(
                        f"{relative_path} still contains pending vectorization "
                        "placeholder text instead of an implemented hot-loop shape."
                    ),
                    file_path=source_path.as_posix(),
                )
            )
        if _looks_like_cross_column_pack(lowered):
            violations.append(
                GeneratedCheckViolation(
                    code="CRITICAL_VECTOR_CROSS_COLUMN_PACK",
                    severity="diagnostic",
                    message=(
                        f"{relative_path} appears to use _mm256_set_pd across "
                        "different usage_* columns; this is not contiguous-row "
                        "hot-loop vectorization evidence."
                    ),
                    file_path=source_path.as_posix(),
                )
            )
    return violations


def _looks_like_pending_vector_placeholder(source_text: str) -> bool:
    """Return True when source keeps vectorization as TODO text."""
    markers = ("todo", "pending", "future", "later")
    vector_markers = ("vector", "simd", "avx")
    return any(marker in source_text for marker in markers) and any(
        marker in source_text for marker in vector_markers
    )


def _looks_like_cross_column_pack(source_text: str) -> bool:
    """Return True for common non-contiguous usage_* AVX packing anti-patterns."""
    return "_mm256_set_pd" in source_text and source_text.count("usage_") >= 2


def _match_antipattern(
    *,
    anti_pattern: str,
    query_id: str,
    file_path: Path,
    source_text: str,
    severity: str,
) -> GeneratedCheckViolation | None:
    """Match one configured anti-pattern against the active query source text."""
    lowered = source_text.lower()
    if anti_pattern == "q6_wrong_formula":
        if "1 - l_discount" in lowered or "1-l_discount" in lowered or "(1-l_discount)" in lowered or "(1 - l_discount)" in lowered:
            return GeneratedCheckViolation(
                code="Q6_WRONG_REVENUE_FORMULA",
                severity="error",
                message=(
                    f"{file_path.name} uses (1-l_discount) pattern — Q6 must use "
                    "raw l_discount directly: sum(l_extendedprice * l_discount), "
                    "NOT sum(l_extendedprice * (1-l_discount)). Do not confuse Q6 with Q1."
                ),
                file_path=file_path.as_posix(),
            )
    return None


def _check_query_family_boundaries(
    *,
    workspace_root: Path,
    active_unit_files: tuple[str, ...],
) -> list[GeneratedCheckViolation]:
    """Validate family-kernel ownership and entrypoint preservation."""
    violations: list[GeneratedCheckViolation] = []
    for relative_path in active_unit_files:
        if not relative_path.startswith("query_family_"):
            continue
        target = workspace_root / relative_path
        if not target.exists():
            violations.append(
                GeneratedCheckViolation(
                    code="QUERY_UNIT_MEMBER_MISSING",
                    severity="diagnostic",
                    message=f"Required family unit file is missing: {relative_path}",
                    file_path=target.as_posix(),
                )
            )
    for file_path in sorted(workspace_root.glob("query_family_*.*")):
        text = file_path.read_text(encoding="utf-8")
        if _EXECUTE_ENTRYPOINT_RE.search(text):
            violations.append(
                GeneratedCheckViolation(
                    code="QUERY_UNIT_ENTRYPOINT_MISSING",
                    severity="error",
                    message=(
                        f"{file_path.name} must not define public execute_q* entrypoints; "
                        "keep entrypoints in query_q*.cpp/.hpp."
                    ),
                    file_path=file_path.as_posix(),
                )
            )
    return violations
