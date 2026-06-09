#pragma once

#include "builder_impl.hpp"
#include <cstddef>
#include <string>

void query(Engine*);

enum class QueryOutputMode {
    FullCsv,
    NoOutput,
    HashOnly,
};

struct QueryResult {
    std::string csv_output;
    bool valid = false;
    bool has_kernel_ms_override = false;
    double kernel_ms_override = 0.0;
    std::size_t row_count = 0;
    std::size_t output_bytes = 0;
};

QueryResult& get_last_query_result();
QueryOutputMode get_query_output_mode();
bool should_materialize_query_output();
[[noreturn]] void raise_missing_template_query_body(const char* query_id);
