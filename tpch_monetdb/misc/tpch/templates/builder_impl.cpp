#include "builder_impl.hpp"
#include "builder_api.hpp"
#include "loader_impl.hpp"

#include <stdexcept>

// Build the query-facing engine state from loader output while preserving the ABI.
Engine* build(RawData* raw_data) {
    if (raw_data == nullptr) {
        throw std::runtime_error("build received null RawData");
    }

    auto* engine = new Engine{};
    engine->is_tpch = raw_data->is_tpch;
    engine->data_path = raw_data->data_path;
    engine->row_count = raw_data->row_count;
    engine->customer_row_count = raw_data->customer_row_count;
    engine->lineitem_row_count = raw_data->lineitem_row_count;
    engine->nation_row_count = raw_data->nation_row_count;
    engine->orders_row_count = raw_data->orders_row_count;
    engine->part_row_count = raw_data->part_row_count;
    engine->partsupp_row_count = raw_data->partsupp_row_count;
    engine->region_row_count = raw_data->region_row_count;
    engine->supplier_row_count = raw_data->supplier_row_count;
    engine->customers = raw_data->customers;
    engine->lineitems = raw_data->lineitems;
    engine->nations = raw_data->nations;
    engine->orders = raw_data->orders;
    engine->parts = raw_data->parts;
    engine->partsupps = raw_data->partsupps;
    engine->regions = raw_data->regions;
    engine->suppliers = raw_data->suppliers;
    return engine;
}
