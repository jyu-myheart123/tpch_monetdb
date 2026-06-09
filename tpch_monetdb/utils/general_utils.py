import textwrap
from pathlib import Path
from typing import Any, Callable, Optional

from tpch_monetdb.dataset.gen_tpch.tpch_queries import (
    TpchQueryContract,
    get_contract as get_tpch_contract,
    list_all_contracts as list_all_tpch_contracts,
)

SUPPORTED_QUERY_IDS = [str(i) for i in range(1, 23)]


def write_query_and_args_file(
    benchmark_name: str,
    gen_placeholders_fn: Callable,
    query_list: list[str],
    out_dir: str,
    use_fasttest_format: bool = True,
    storage_plan: Optional[str] = None,
) -> str:
    """生成 queries.txt 与 args_parser.hpp 的 query artifact 入口.

    phase10 起 bootstrap 链不再向 `query_impl.cpp` 注入 example parser 代码；
    dispatcher 源码由模板与模型维护，args_parser.hpp 与 queries.txt 为唯一生成目标。
    """
    normalized_benchmark = benchmark_name.lower()
    if normalized_benchmark != "tpch":
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    query_file = out_path / "queries.txt"
    args_file = out_path / "args_parser.hpp"

    sql_template_list = []
    normalized_query_list = [
        _normalize_tpch_query_id_for_artifact(query_id)
        for query_id in query_list
    ]
    for query_id in normalized_query_list:
        contract = get_tpch_contract(query_id)
        sql_template_list.append(
            _render_tpch_query_contract_for_queries_txt(query_id, contract)
        )
    args_str, example_code = gen_tpch_args_str(
        normalized_query_list,
        use_fasttest_format=use_fasttest_format,
        gen_placeholders_fn=gen_placeholders_fn,
    )
    qf_string = "\n\n".join(sql_template_list)
    query_file.write_text(qf_string)

    if use_fasttest_format:
        args_file.write_text(args_str)
    else:
        args_file.write_text(f"{args_str}\n{example_code}")

    folder_context = f"{qf_string}\n\n{args_str}"
    if storage_plan is not None:
        storage_plan_file = out_path / "storage_plan.txt"
        if storage_plan_file.exists():
            existing = storage_plan_file.read_text()
            assert existing == storage_plan, (
                f"Storage plan file already exists at {storage_plan_file} with different contents."
            )
        else:
            storage_plan_file.write_text(storage_plan)
        folder_context += f"\n\n{storage_plan}"
    return folder_context


def _normalize_tpch_query_id_for_artifact(query_id: str) -> str:
    """Normalize query ids used in TPC-H query artifacts to Q-prefixed form."""
    return get_tpch_contract(query_id).query_id


def _render_tpch_query_contract_for_queries_txt(
    query_id: str,
    contract: TpchQueryContract,
) -> str:
    """Render a TPC-H contract into an agent-readable queries.txt section."""
    lines = [
        f"Query {query_id}:",
        "Benchmark: TPC-H",
        f"Tables: {', '.join(contract.tables)}",
        f"Features: {', '.join(contract.features)}",
        f"Parameters: {', '.join(contract.parameter_names) or 'none'}",
        f"Result ordering: {contract.ordering.strategy}",
        f"Result ordered: {str(contract.result_ordered).lower()}",
        f"Comparison: {contract.comparison.strategy}",
        f"Float tolerance: atol={contract.float_atol}, rtol={contract.float_rtol}",
        f"Container profile: {contract.container_profile}",
        f"Dialect notes: {contract.dialect_notes}",
    ]
    if contract.sorted_by:
        lines.append(f"Ordering: {', '.join(contract.sorted_by)}")
    lines.append("SQL template:")
    lines.append(contract.sql_template)
    return "\n".join(lines)


def _cpp_type_for_tpch_placeholder(value: Any) -> str:
    """Return a conservative C++ field type for a sampled TPC-H placeholder."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    return "std::string"


def _cpp_tpch_assignment(placeholder: str, value: Any, query_id: str) -> str:
    """Return C++ code that assigns one TPC-H key=value placeholder."""
    required = f'require_arg(kv, "{placeholder}", "{query_id}")'
    if isinstance(value, bool):
        return f'    args.{placeholder} = ({required} == "true");\n'
    if isinstance(value, int):
        return f"    args.{placeholder} = std::stoi({required});\n"
    if isinstance(value, float):
        return f"    args.{placeholder} = std::stod({required});\n"
    return f"    args.{placeholder} = {required};\n"


def gen_tpch_args_str(
    query_ids: list[str],
    gen_placeholders_fn: Callable,
    use_fasttest_format: bool = True,
) -> tuple[str, str]:
    """Generate a TPC-H args_parser.hpp that consumes QID KEY=value lines."""
    if not use_fasttest_format:
        raise Exception("Non-fasttest format is outdated and no longer supported.")

    out_str = """#pragma once

#include <cctype>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>

struct QueryRequest {
    std::string id;
    std::string line;
};

inline std::unordered_map<std::string, std::string> parse_key_value_args(std::istringstream& iss) {
    std::unordered_map<std::string, std::string> result;
    std::string text;
    std::getline(iss, text);
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos >= text.size()) {
            break;
        }
        const size_t key_start = pos;
        while (pos < text.size() && text[pos] != '=' && !std::isspace(static_cast<unsigned char>(text[pos]))) {
            ++pos;
        }
        if (pos >= text.size() || text[pos] != '=') {
            throw std::runtime_error("Expected KEY=value argument");
        }
        const std::string key = text.substr(key_start, pos - key_start);
        ++pos;
        std::string value;
        if (pos < text.size() && text[pos] == '"') {
            ++pos;
            while (pos < text.size()) {
                const char ch = text[pos++];
                if (ch == '\\\\' && pos < text.size()) {
                    value.push_back(text[pos++]);
                    continue;
                }
                if (ch == '"') {
                    break;
                }
                value.push_back(ch);
            }
        } else {
            const size_t value_start = pos;
            while (pos < text.size() && !std::isspace(static_cast<unsigned char>(text[pos]))) {
                ++pos;
            }
            value = text.substr(value_start, pos - value_start);
        }
        result[key] = value;
    }
    return result;
}

inline const std::string& require_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::string& query_id
) {
    const auto it = args.find(key);
    if (it == args.end()) {
        throw std::runtime_error(query_id + ": missing required argument " + key);
    }
    return it->second;
}

"""
    generated_query_ids = list_all_tpch_contracts()
    requested_query_ids = set(query_ids)
    for query_id in generated_query_ids:
        placeholders_dict = gen_placeholders_fn(query_name=query_id)
        placeholder_lines = [
            f"    {_cpp_type_for_tpch_placeholder(value)} {placeholder};"
            for placeholder, value in placeholders_dict.items()
        ]
        placeholder_str = "\n".join(placeholder_lines)
        out_str += f"\n//{query_id}\nstruct {query_id}Args {{\n{placeholder_str}\n}};\n"
        out_str += f"inline {query_id}Args parse_{query_id.lower()}(const QueryRequest& request) {{\n"
        out_str += f"    {query_id}Args args;\n"
        out_str += "    std::istringstream iss(request.line);\n"
        out_str += "    std::string qid;\n"
        out_str += "    if (!(iss >> qid)) {\n"
        out_str += f'        throw std::runtime_error("{query_id}: failed to parse query id");\n'
        out_str += "    }\n"
        out_str += "    const auto kv = parse_key_value_args(iss);\n"
        for placeholder, value in placeholders_dict.items():
            out_str += _cpp_tpch_assignment(placeholder, value, query_id)
        out_str += "    return args;\n}\n"

    example_code = "\n// Example code for requested TPC-H parse functions:\n"
    for query_id in query_ids:
        if query_id in requested_query_ids:
            example_code += f"// {query_id}: parse_{query_id.lower()}(request)\n"
    return out_str, example_code


def get_affinity_prompt(
    include_numa: bool = False,
    filename: str = "cpu_affinity.hpp",
) -> str:
    numa_section = ""
    if include_numa:
        numa_section = textwrap.dedent(
            """\
            NUMA placement:
              Pin the current process to a specific NUMA node to improve memory locality
              during initialization or data ingestion:
                void pin_process_to_numa_node(int node_id);

              Query the number of logical CPUs associated with a NUMA node:
                int get_numa_node_cpu_count(int node_id);

        """
        )

    return textwrap.dedent(
        f"""\
        CPU affinity helpers is predefined in {filename}.
        You have to use the following functions, no need to implement them yourself,
        they are already provided by the runtime:

        {numa_section}CPU affinity:
          Pin the process to a single logical CPU for deterministic execution:
            void pin_process_to_cpu(int cpu_id);

          Restore affinity to all available CPUs:
            void unpin_process_from_cpus();
    """
    )
