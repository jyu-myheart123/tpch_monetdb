#pragma once

#include "loader_impl.hpp"
#include <cstddef>
#include <string>
#include <vector>

struct Engine {
    bool is_tpch = false;
    std::string data_path;
    std::size_t row_count = 0;
    std::size_t customer_row_count = 0;
    std::size_t lineitem_row_count = 0;
    std::size_t nation_row_count = 0;
    std::size_t orders_row_count = 0;
    std::size_t part_row_count = 0;
    std::size_t partsupp_row_count = 0;
    std::size_t region_row_count = 0;
    std::size_t supplier_row_count = 0;
    std::vector<CustomerRow> customers;
    std::vector<LineitemRow> lineitems;
    std::vector<NationRow> nations;
    std::vector<OrdersRow> orders;
    std::vector<PartRow> parts;
    std::vector<PartsuppRow> partsupps;
    std::vector<RegionRow> regions;
    std::vector<SupplierRow> suppliers;
};

Engine* build(RawData*);
