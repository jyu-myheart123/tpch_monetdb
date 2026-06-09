#include "query_impl.hpp"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "query_registry_generated.hpp"

namespace {
QueryResult g_last_query_result;
}

QueryResult& get_last_query_result() {
    return g_last_query_result;
}

// Return the process-wide query output mode requested by the Python harness.
QueryOutputMode get_query_output_mode() {
    static const QueryOutputMode mode = []() {
        const char* raw_mode = std::getenv("TPCH_MONETDB_QUERY_OUTPUT_MODE");
        if (raw_mode == nullptr) {
            return QueryOutputMode::FullCsv;
        }
        const std::string mode_text(raw_mode);
        if (mode_text == "no_output") {
            return QueryOutputMode::NoOutput;
        }
        if (mode_text == "hash_only") {
            return QueryOutputMode::HashOnly;
        }
        return QueryOutputMode::FullCsv;
    }();
    return mode;
}

// Tell generated query code whether it should build the full CSV payload.
bool should_materialize_query_output() {
    return get_query_output_mode() == QueryOutputMode::FullCsv;
}

// Query templates intentionally carry no TPC-H algorithm body.
[[noreturn]] void raise_missing_template_query_body(const char* query_id) {
    throw std::runtime_error(
        std::string("Template query body is absent for ") + query_id
    );
}

void query(Engine* engine) {
    std::vector<QueryRequest> requests;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            break;
        }
        std::istringstream iss(line);
        std::string query_id = "0";
        iss >> query_id;
        if (!iss) {
            continue;
        }
        requests.push_back(QueryRequest{query_id, line});
    }

    for (size_t run_nr = 0; run_nr < requests.size(); ++run_nr) {
        auto& x = requests[run_nr];
        g_last_query_result.csv_output.clear();
        g_last_query_result.valid = false;
        g_last_query_result.has_kernel_ms_override = false;
        g_last_query_result.kernel_ms_override = 0.0;
        g_last_query_result.row_count = 0;
        g_last_query_result.output_bytes = 0;

        const auto query_t0 = std::chrono::steady_clock::now();
        const QueryRequest request{x.id, x.line};
        const auto kernel_t0 = std::chrono::steady_clock::now();

        dispatch_query(*engine, request);

        const auto kernel_t1 = std::chrono::steady_clock::now();
        const auto query_t1 = std::chrono::steady_clock::now();
        const double measured_kernel_ms =
            std::chrono::duration<double, std::milli>(kernel_t1 - kernel_t0).count();
        const double kernel_ms = g_last_query_result.has_kernel_ms_override
            ? g_last_query_result.kernel_ms_override
            : measured_kernel_ms;
        const double query_ms =
            std::chrono::duration<double, std::milli>(query_t1 - query_t0).count();

        std::printf("%s | Execution ms: %.3f\n", x.id.c_str(), kernel_ms);
        std::printf("%s | Query ms: %.3f\n", x.id.c_str(), query_ms);

        if (should_materialize_query_output() && g_last_query_result.valid) {
            g_last_query_result.output_bytes = g_last_query_result.csv_output.size();
            const std::string output_path =
                "result" + std::to_string(run_nr + 1) + ".csv";
            if (std::FILE* fp = std::fopen(output_path.c_str(), "w")) {
                std::fwrite(
                    g_last_query_result.csv_output.data(),
                    1,
                    g_last_query_result.csv_output.size(),
                    fp
                );
                std::fclose(fp);
            }
        }
    }
}
