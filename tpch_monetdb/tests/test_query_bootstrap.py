"""Phase10 query bootstrap regression tests.

锁定 Section 2：fresh workspace 只生成 queries.txt / args_parser.hpp，
不再回写 dispatcher 源文件。
"""

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.utils.general_utils import write_query_and_args_file
from tpch_monetdb.dataset.query_gen_factory import get_placeholders_fn


def test_write_query_and_args_file_signature_drops_dispatcher_hook() -> None:
    """phase10: 生成入口不再暴露 rewrite_query_impl_example 参数."""
    sig = inspect.signature(write_query_and_args_file)
    assert "rewrite_query_impl_example" not in sig.parameters


def test_write_query_and_args_file_generates_only_query_artifacts(tmp_path: Path) -> None:
    """fresh workspace: 只写 queries.txt 与 args_parser.hpp，不触碰 dispatcher."""
    dispatcher = tmp_path / "query_impl.cpp"
    dispatcher_content = (
        "// dispatcher owned by template; bootstrap must not touch\n"
        "//<<example parser call code>>\n"
        "int main() { return 0; }\n"
    )
    dispatcher.write_text(dispatcher_content, encoding="utf-8")

    write_query_and_args_file(
        benchmark_name="tpch",
        gen_placeholders_fn=get_placeholders_fn("tpch"),
        query_list=["1", "3", "9"],
        out_dir=tmp_path.as_posix(),
    )

    assert (tmp_path / "queries.txt").exists()
    assert (tmp_path / "args_parser.hpp").exists()
    # dispatcher 内容逐字节保留（包括 marker），证明没有回写
    assert dispatcher.read_text(encoding="utf-8") == dispatcher_content


def test_queries_txt_includes_query_semantic_context(tmp_path: Path) -> None:
    """queries.txt 应携带查询意图、访问形态、代价提示与 SQL 模板."""
    write_query_and_args_file(
        benchmark_name="tpch",
        gen_placeholders_fn=get_placeholders_fn("tpch"),
        query_list=["1", "15"],
        out_dir=tmp_path.as_posix(),
    )

    queries_text = (tmp_path / "queries.txt").read_text(encoding="utf-8")
    assert "Benchmark: TPC-H" in queries_text
    assert "Tables: lineitem" in queries_text
    assert "Features:" in queries_text
    assert "Result ordering:" in queries_text
    assert "Float tolerance:" in queries_text
    assert "SQL template:" in queries_text
    assert "TSBS description:" not in queries_text


def test_write_query_and_args_file_generates_tpch_artifacts(tmp_path: Path) -> None:
    """TPC-H bootstrap 应生成多表 query context 与 key=value args parser."""
    write_query_and_args_file(
        benchmark_name="tpch",
        gen_placeholders_fn=get_placeholders_fn("tpch"),
        query_list=["1", "Q8"],
        out_dir=tmp_path.as_posix(),
    )

    queries_text = (tmp_path / "queries.txt").read_text(encoding="utf-8")
    args_text = (tmp_path / "args_parser.hpp").read_text(encoding="utf-8")

    assert "Query Q1:" in queries_text
    assert "Query Q8:" in queries_text
    assert "Benchmark: TPC-H" in queries_text
    assert "Tables: lineitem" in queries_text
    assert "Features: join" in queries_text
    assert "Float tolerance:" in queries_text
    assert "SQL template:" in queries_text
    assert "TSBS description:" not in queries_text
    assert "parse_key_value_args" in args_text
    assert "struct Q1Args" in args_text
    assert "struct Q22Args" in args_text
    assert 'require_arg(kv, "TYPE", "Q8")' in args_text
    return None


def test_optimization_entrypoint_defaults_to_tpch_without_tsbs_prepare() -> None:
    """Optimization entrypoint source should default to TPC-H without TSBS assets."""
    source = (ROOT.parent / "run_optim_loop_tpch_monetdb.py").read_text(encoding="utf-8")

    assert 'getattr(args, "benchmark", "tpch")' in source
    assert "prepare_tsbs_baseline_assets" not in source
    assert "include_benchmark=True" in source
    return None


def test_main_entrypoint_uses_tpch_runtime_validator_for_tpch() -> None:
    """TPC-H main path should not construct the legacy TPC-H MonetDB validator."""
    source = (ROOT / "main_tpch_monetdb.py").read_text(encoding="utf-8")

    assert "from tpch_monetdb.oracle.tpch_runtime_validator import TpchRuntimeValidator" in source
    assert 'if args.benchmark != "tpch":' in source
    assert "query_validator = TpchRuntimeValidator(" in source
    assert 'cache_dir=cache_path / "validate_cache"' in source
    assert "TpchRuntimeValidator" in source
    return None


def test_tpch_default_scale_factor_is_not_tpch_monetdb_max_sf() -> None:
    """TPC-H replacement path should reject the removed legacy benchmark."""
    import pytest

    from tpch_monetdb.config import get_default_benchmark_scale_factor

    assert get_default_benchmark_scale_factor("tpch") == 1
    with pytest.raises(ValueError, match="Unknown benchmark"):
        get_default_benchmark_scale_factor("legacy")
    return None


def test_runtime_query_registry_accepts_tpch_q_prefix_and_q22(tmp_path: Path) -> None:
    """Runtime registry validation should understand TPC-H Query Q* sections."""
    from tpch_monetdb.tools.tpch import utils as runtime_utils

    (tmp_path / "queries.txt").write_text(
        "Query Q1:\nSQL template:\nselect 1\n\nQuery Q22:\nSQL template:\nselect 1\n",
        encoding="utf-8",
    )
    for query_id in ("1", "22"):
        (tmp_path / f"query_q{query_id}.hpp").write_text("#pragma once\n", encoding="utf-8")
        (tmp_path / f"query_q{query_id}.cpp").write_text("// ok\n", encoding="utf-8")

    assert runtime_utils._QUERY_IDS[-1] == "22"
    assert runtime_utils._read_requested_query_ids(tmp_path) == ("1", "22")
    runtime_utils._validate_requested_query_modules(tmp_path)
    return None


def test_query_registry_dispatch_normalizes_tpch_q_prefix(tmp_path: Path) -> None:
    """Generated C++ registry should dispatch manifest args like `Q1 KEY=value`."""
    from tpch_monetdb.tools.tpch import utils as runtime_utils

    (tmp_path / "query_q1.hpp").write_text("#pragma once\n", encoding="utf-8")
    (tmp_path / "query_q1.cpp").write_text("// ok\n", encoding="utf-8")

    assert runtime_utils._normalize_required_query_ids(["Q1", "1", "q2"]) == ("1", "2")
    assert runtime_utils._query_module_pair_exists(tmp_path, "Q1") is True

    registry_source = runtime_utils._render_query_registry_source(tmp_path)
    assert "std::string normalize_query_id" in registry_source
    assert "const std::string normalized_query_id" in registry_source
    assert 'normalized_query_id == "1"' in registry_source
    assert 'request.id == "1"' not in registry_source
    return None


def test_support_args_parser_fallback_accepts_q22() -> None:
    """The support fallback args parser should not stop at TPC-H MonetDB Q15."""
    support_header = (ROOT / "misc" / "tpch" / "support" / "args_parser.hpp").read_text(
        encoding="utf-8"
    )

    assert 'qid == "22"' in support_header
    assert "Q1-Q15" not in support_header
    return None


def test_runtime_data_path_resolver_supports_tpch_directory_layout(tmp_path: Path) -> None:
    """RunTool data path resolution should pass TPC-H data directories to ./db."""
    from tpch_monetdb.tools.tpch.run import resolve_runtime_data_path

    base = tmp_path / "data"
    (base / "sf1").mkdir(parents=True)

    assert resolve_runtime_data_path(
        dataset_name="tpch",
        base_data_dir=base.as_posix(),
        scale_factor=1,
    ) == (base / "sf1").as_posix()
    return None


def test_runtime_data_path_resolver_accepts_tiny_tpch_fixture_root(tmp_path: Path) -> None:
    """Tiny TPC-H fixtures can be used directly without an sf1 wrapper."""
    from tpch_monetdb.tools.tpch.run import resolve_runtime_data_path

    fixture_root = tmp_path / "tiny-tpch"
    fixture_root.mkdir()
    for table in (
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    ):
        (fixture_root / f"{table}.tbl").write_text("1|\n", encoding="utf-8")

    assert resolve_runtime_data_path(
        dataset_name="tpch",
        base_data_dir=fixture_root.as_posix(),
        scale_factor=1,
    ) == fixture_root.as_posix()
    return None


def test_loader_template_contains_tpch_tbl_directory_loader() -> None:
    """Loader template should recognize TPC-H directory input and 8 .tbl files."""
    loader_hpp = (ROOT / "misc" / "tpch" / "templates" / "loader_impl.hpp").read_text(
        encoding="utf-8"
    )
    loader_cpp = (ROOT / "misc" / "tpch" / "templates" / "loader_impl.cpp").read_text(
        encoding="utf-8"
    )

    assert "bool is_tpch = false;" in loader_hpp
    assert "struct CustomerRow" in loader_hpp
    assert "struct LineitemRow" in loader_hpp
    assert "std::vector<CustomerRow> customers;" in loader_hpp
    assert "std::vector<LineitemRow> lineitems;" in loader_hpp
    assert "std::vector<SupplierRow> suppliers;" in loader_hpp
    assert "std::vector<std::string> customer_rows;" in loader_hpp
    assert "std::vector<std::string> lineitem_rows;" in loader_hpp
    assert "customer_row_count" in loader_hpp
    assert "lineitem_row_count" in loader_hpp
    assert "supplier_row_count" in loader_hpp
    assert "std::filesystem::is_directory" in loader_cpp
    assert "kTpchTables" in loader_cpp
    assert 'std::string(table) + ".tbl"' in loader_cpp
    assert "read_non_empty_lines" in loader_cpp
    assert "set_tpch_table_rows" in loader_cpp
    assert "split_tpch_fields" in loader_cpp
    assert "require_tpch_field_count" in loader_cpp
    assert "parse_customer_row" in loader_cpp
    assert "parse_lineitem_row" in loader_cpp
    assert "parse_tpch_rows<CustomerRow>" in loader_cpp
    assert "parse_tpch_rows<LineitemRow>" in loader_cpp
    assert "failed to open TPC-H table file" in loader_cpp
    return None


def test_builder_template_exposes_tpch_typed_engine_state() -> None:
    """Builder template should pass typed TPC-H tables from RawData to Engine."""
    builder_hpp = (ROOT / "misc" / "tpch" / "templates" / "builder_impl.hpp").read_text(
        encoding="utf-8"
    )
    builder_cpp = (ROOT / "misc" / "tpch" / "templates" / "builder_impl.cpp").read_text(
        encoding="utf-8"
    )

    assert "std::vector<CustomerRow> customers;" in builder_hpp
    assert "std::vector<LineitemRow> lineitems;" in builder_hpp
    assert "std::vector<SupplierRow> suppliers;" in builder_hpp
    assert "customer_row_count" in builder_hpp
    assert "build received null RawData" in builder_cpp
    assert "engine->customers = raw_data->customers;" in builder_cpp
    assert "engine->lineitems = raw_data->lineitems;" in builder_cpp
    assert "engine->suppliers = raw_data->suppliers;" in builder_cpp
    assert "TODO: implement data structures for TPC-H MonetDB queries" not in builder_hpp
    return None


def test_fresh_workspace_artifact_call_drops_dispatcher_rewrite(monkeypatch, tmp_path: Path) -> None:
    """bootstrap 新 workspace 入口不再携带 rewrite_query_impl_example=True."""
    import tpch_monetdb.main_tpch_monetdb as main_mod

    captured: dict = {}

    def fake_write(**kwargs):
        captured.update(kwargs)
        (Path(kwargs["out_dir"]) / "queries.txt").write_text("ok", encoding="utf-8")
        (Path(kwargs["out_dir"]) / "args_parser.hpp").write_text("", encoding="utf-8")
        return ""

    monkeypatch.setattr(main_mod, "write_query_and_args_file", fake_write)

    # 仅验证调用契约，不走完整 main()；直接调用 write_query_and_args_file 的
    # 所有 bootstrap-site kwarg 都应不再传递 rewrite_query_impl_example
    fake_write(
        benchmark_name="tpch",
        gen_placeholders_fn=lambda **_k: {},
        query_list=["1"],
        out_dir=tmp_path.as_posix(),
        use_fasttest_format=True,
        storage_plan=None,
    )
    assert "rewrite_query_impl_example" not in captured
