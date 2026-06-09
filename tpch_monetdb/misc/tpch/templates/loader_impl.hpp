#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct CustomerRow {
    int c_custkey = 0;
    std::string c_name;
    std::string c_address;
    int c_nationkey = 0;
    std::string c_phone;
    double c_acctbal = 0.0;
    std::string c_mktsegment;
    std::string c_comment;
};

struct LineitemRow {
    int l_orderkey = 0;
    int l_partkey = 0;
    int l_suppkey = 0;
    int l_linenumber = 0;
    double l_quantity = 0.0;
    double l_extendedprice = 0.0;
    double l_discount = 0.0;
    double l_tax = 0.0;
    std::string l_returnflag;
    std::string l_linestatus;
    std::string l_shipdate;
    std::string l_commitdate;
    std::string l_receiptdate;
    std::string l_shipinstruct;
    std::string l_shipmode;
    std::string l_comment;
};

struct NationRow {
    int n_nationkey = 0;
    std::string n_name;
    int n_regionkey = 0;
    std::string n_comment;
};

struct OrdersRow {
    int o_orderkey = 0;
    int o_custkey = 0;
    std::string o_orderstatus;
    double o_totalprice = 0.0;
    std::string o_orderdate;
    std::string o_orderpriority;
    std::string o_clerk;
    int o_shippriority = 0;
    std::string o_comment;
};

struct PartRow {
    int p_partkey = 0;
    std::string p_name;
    std::string p_mfgr;
    std::string p_brand;
    std::string p_type;
    int p_size = 0;
    std::string p_container;
    double p_retailprice = 0.0;
    std::string p_comment;
};

struct PartsuppRow {
    int ps_partkey = 0;
    int ps_suppkey = 0;
    int ps_availqty = 0;
    double ps_supplycost = 0.0;
    std::string ps_comment;
};

struct RegionRow {
    int r_regionkey = 0;
    std::string r_name;
    std::string r_comment;
};

struct SupplierRow {
    int s_suppkey = 0;
    std::string s_name;
    std::string s_address;
    int s_nationkey = 0;
    std::string s_phone;
    double s_acctbal = 0.0;
    std::string s_comment;
};

struct RawData {
    // start: table-defs
    // end: table-defs

    bool is_tpch = false;
    std::string data_path;
    std::vector<std::string> customer_rows;
    std::vector<std::string> lineitem_rows;
    std::vector<std::string> nation_rows;
    std::vector<std::string> orders_rows;
    std::vector<std::string> part_rows;
    std::vector<std::string> partsupp_rows;
    std::vector<std::string> region_rows;
    std::vector<std::string> supplier_rows;
    std::vector<CustomerRow> customers;
    std::vector<LineitemRow> lineitems;
    std::vector<NationRow> nations;
    std::vector<OrdersRow> orders;
    std::vector<PartRow> parts;
    std::vector<PartsuppRow> partsupps;
    std::vector<RegionRow> regions;
    std::vector<SupplierRow> suppliers;
    std::size_t customer_row_count = 0;
    std::size_t lineitem_row_count = 0;
    std::size_t nation_row_count = 0;
    std::size_t orders_row_count = 0;
    std::size_t part_row_count = 0;
    std::size_t partsupp_row_count = 0;
    std::size_t region_row_count = 0;
    std::size_t supplier_row_count = 0;
    std::size_t row_count = 0;
};

RawData* load(std::string);
