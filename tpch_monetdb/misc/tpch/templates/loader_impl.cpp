#include "loader_impl.hpp"
#include "loader_api.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

double parse_ilp_metric_value(std::string token) {
    if (!token.empty() && token.back() == 'i') {
        token.pop_back();
    }
    return std::stod(token);
}

void append_csv_field_escaped(std::string& out, std::string_view field) {
    const bool needs_quotes =
        field.find(',') != std::string_view::npos
        || field.find('"') != std::string_view::npos
        || field.find('\n') != std::string_view::npos
        || field.find('\r') != std::string_view::npos;
    if (!needs_quotes) {
        out.append(field);
        return;
    }
    out.push_back('"');
    for (const char ch : field) {
        if (ch == '"') {
            out.push_back('"');
        }
        out.push_back(ch);
    }
    out.push_back('"');
}

// Split one TPC-H `.tbl` line and normalize the trailing delimiter emitted by dbgen.
std::vector<std::string> split_tpch_fields(std::string_view line) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (std::size_t index = 0; index < line.size(); ++index) {
        if (line[index] == '|') {
            fields.emplace_back(line.substr(start, index - start));
            start = index + 1;
        }
    }
    fields.emplace_back(line.substr(start));
    if (!fields.empty() && fields.back().empty()) {
        fields.pop_back();
    }
    return fields;
}

// Validate that a `.tbl` row matches the TPC-H schema before typed parsing.
void require_tpch_field_count(
    std::string_view table,
    const std::vector<std::string>& fields,
    std::size_t expected
) {
    if (fields.size() != expected) {
        throw std::runtime_error(
            "invalid TPC-H row for table "
            + std::string(table)
            + ": expected "
            + std::to_string(expected)
            + " fields, got "
            + std::to_string(fields.size())
        );
    }
    return;
}

// Parse an integer field and include table context in diagnostics.
int parse_int_field(
    std::string_view table,
    const std::vector<std::string>& fields,
    std::size_t index
) {
    try {
        std::size_t parsed = 0;
        const int result = std::stoi(fields.at(index), &parsed);
        if (parsed != fields.at(index).size()) {
            throw std::invalid_argument("trailing integer characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid integer field "
            + std::to_string(index)
            + " for TPC-H table "
            + std::string(table)
            + ": "
            + fields.at(index)
        );
    }
}

// Parse a decimal field and include table context in diagnostics.
double parse_double_field(
    std::string_view table,
    const std::vector<std::string>& fields,
    std::size_t index
) {
    try {
        std::size_t parsed = 0;
        const double result = std::stod(fields.at(index), &parsed);
        if (parsed != fields.at(index).size()) {
            throw std::invalid_argument("trailing decimal characters");
        }
        return result;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid decimal field "
            + std::to_string(index)
            + " for TPC-H table "
            + std::string(table)
            + ": "
            + fields.at(index)
        );
    }
}

// Parse one customer row according to the TPC-H customer schema.
CustomerRow parse_customer_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("customer", fields, 8);
    CustomerRow row{};
    row.c_custkey = parse_int_field("customer", fields, 0);
    row.c_name = fields[1];
    row.c_address = fields[2];
    row.c_nationkey = parse_int_field("customer", fields, 3);
    row.c_phone = fields[4];
    row.c_acctbal = parse_double_field("customer", fields, 5);
    row.c_mktsegment = fields[6];
    row.c_comment = fields[7];
    return row;
}

// Parse one lineitem row according to the TPC-H lineitem schema.
LineitemRow parse_lineitem_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("lineitem", fields, 16);
    LineitemRow row{};
    row.l_orderkey = parse_int_field("lineitem", fields, 0);
    row.l_partkey = parse_int_field("lineitem", fields, 1);
    row.l_suppkey = parse_int_field("lineitem", fields, 2);
    row.l_linenumber = parse_int_field("lineitem", fields, 3);
    row.l_quantity = parse_double_field("lineitem", fields, 4);
    row.l_extendedprice = parse_double_field("lineitem", fields, 5);
    row.l_discount = parse_double_field("lineitem", fields, 6);
    row.l_tax = parse_double_field("lineitem", fields, 7);
    row.l_returnflag = fields[8];
    row.l_linestatus = fields[9];
    row.l_shipdate = fields[10];
    row.l_commitdate = fields[11];
    row.l_receiptdate = fields[12];
    row.l_shipinstruct = fields[13];
    row.l_shipmode = fields[14];
    row.l_comment = fields[15];
    return row;
}

// Parse one nation row according to the TPC-H nation schema.
NationRow parse_nation_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("nation", fields, 4);
    NationRow row{};
    row.n_nationkey = parse_int_field("nation", fields, 0);
    row.n_name = fields[1];
    row.n_regionkey = parse_int_field("nation", fields, 2);
    row.n_comment = fields[3];
    return row;
}

// Parse one orders row according to the TPC-H orders schema.
OrdersRow parse_orders_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("orders", fields, 9);
    OrdersRow row{};
    row.o_orderkey = parse_int_field("orders", fields, 0);
    row.o_custkey = parse_int_field("orders", fields, 1);
    row.o_orderstatus = fields[2];
    row.o_totalprice = parse_double_field("orders", fields, 3);
    row.o_orderdate = fields[4];
    row.o_orderpriority = fields[5];
    row.o_clerk = fields[6];
    row.o_shippriority = parse_int_field("orders", fields, 7);
    row.o_comment = fields[8];
    return row;
}

// Parse one part row according to the TPC-H part schema.
PartRow parse_part_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("part", fields, 9);
    PartRow row{};
    row.p_partkey = parse_int_field("part", fields, 0);
    row.p_name = fields[1];
    row.p_mfgr = fields[2];
    row.p_brand = fields[3];
    row.p_type = fields[4];
    row.p_size = parse_int_field("part", fields, 5);
    row.p_container = fields[6];
    row.p_retailprice = parse_double_field("part", fields, 7);
    row.p_comment = fields[8];
    return row;
}

// Parse one partsupp row according to the TPC-H partsupp schema.
PartsuppRow parse_partsupp_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("partsupp", fields, 5);
    PartsuppRow row{};
    row.ps_partkey = parse_int_field("partsupp", fields, 0);
    row.ps_suppkey = parse_int_field("partsupp", fields, 1);
    row.ps_availqty = parse_int_field("partsupp", fields, 2);
    row.ps_supplycost = parse_double_field("partsupp", fields, 3);
    row.ps_comment = fields[4];
    return row;
}

// Parse one region row according to the TPC-H region schema.
RegionRow parse_region_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("region", fields, 3);
    RegionRow row{};
    row.r_regionkey = parse_int_field("region", fields, 0);
    row.r_name = fields[1];
    row.r_comment = fields[2];
    return row;
}

// Parse one supplier row according to the TPC-H supplier schema.
SupplierRow parse_supplier_row(std::string_view line) {
    const auto fields = split_tpch_fields(line);
    require_tpch_field_count("supplier", fields, 7);
    SupplierRow row{};
    row.s_suppkey = parse_int_field("supplier", fields, 0);
    row.s_name = fields[1];
    row.s_address = fields[2];
    row.s_nationkey = parse_int_field("supplier", fields, 3);
    row.s_phone = fields[4];
    row.s_acctbal = parse_double_field("supplier", fields, 5);
    row.s_comment = fields[6];
    return row;
}

// Parse all rows in a TPC-H table with a supplied row parser.
template <typename Row, typename Parser>
std::vector<Row> parse_tpch_rows(
    const std::vector<std::string>& rows,
    Parser parser
) {
    std::vector<Row> parsed_rows;
    parsed_rows.reserve(rows.size());
    for (const std::string& row : rows) {
        parsed_rows.push_back(parser(row));
    }
    return parsed_rows;
}

std::vector<std::string> read_non_empty_lines(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open TPC-H table file: " + path.string());
    }
    std::vector<std::string> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            rows.push_back(line);
        }
    }
    return rows;
}

void set_tpch_table_rows(
    RawData* raw,
    std::string_view table,
    std::vector<std::string> rows
) {
    const std::size_t row_count = rows.size();
    if (table == "customer") {
        raw->customer_rows = std::move(rows);
        raw->customers = parse_tpch_rows<CustomerRow>(
            raw->customer_rows,
            parse_customer_row
        );
        raw->customer_row_count = row_count;
    } else if (table == "lineitem") {
        raw->lineitem_rows = std::move(rows);
        raw->lineitems = parse_tpch_rows<LineitemRow>(
            raw->lineitem_rows,
            parse_lineitem_row
        );
        raw->lineitem_row_count = row_count;
    } else if (table == "nation") {
        raw->nation_rows = std::move(rows);
        raw->nations = parse_tpch_rows<NationRow>(
            raw->nation_rows,
            parse_nation_row
        );
        raw->nation_row_count = row_count;
    } else if (table == "orders") {
        raw->orders_rows = std::move(rows);
        raw->orders = parse_tpch_rows<OrdersRow>(
            raw->orders_rows,
            parse_orders_row
        );
        raw->orders_row_count = row_count;
    } else if (table == "part") {
        raw->part_rows = std::move(rows);
        raw->parts = parse_tpch_rows<PartRow>(
            raw->part_rows,
            parse_part_row
        );
        raw->part_row_count = row_count;
    } else if (table == "partsupp") {
        raw->partsupp_rows = std::move(rows);
        raw->partsupps = parse_tpch_rows<PartsuppRow>(
            raw->partsupp_rows,
            parse_partsupp_row
        );
        raw->partsupp_row_count = row_count;
    } else if (table == "region") {
        raw->region_rows = std::move(rows);
        raw->regions = parse_tpch_rows<RegionRow>(
            raw->region_rows,
            parse_region_row
        );
        raw->region_row_count = row_count;
    } else if (table == "supplier") {
        raw->supplier_rows = std::move(rows);
        raw->suppliers = parse_tpch_rows<SupplierRow>(
            raw->supplier_rows,
            parse_supplier_row
        );
        raw->supplier_row_count = row_count;
    }
}

}

RawData* load(std::string path) {
    auto* raw = new RawData{};
    raw->data_path = path;
    const std::filesystem::path input_path(path);
    if (std::filesystem::is_directory(input_path)) {
        raw->is_tpch = true;
        constexpr std::array<std::string_view, 8> kTpchTables = {
            "customer",
            "lineitem",
            "nation",
            "orders",
            "part",
            "partsupp",
            "region",
            "supplier",
        };
        for (const std::string_view table : kTpchTables) {
            const std::filesystem::path table_path =
                input_path / (std::string(table) + ".tbl");
            auto table_rows = read_non_empty_lines(table_path);
            raw->row_count += table_rows.size();
            set_tpch_table_rows(raw, table, std::move(table_rows));
        }

        // start: table-reads
        // end: table-reads

        return raw;
    }

    std::ifstream input(input_path);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            ++raw->row_count;
        }
    }

    // start: table-reads
    // end: table-reads

    return raw;
}
