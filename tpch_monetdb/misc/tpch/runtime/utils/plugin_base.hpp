#pragma once

// Plugin base definitions
// This file is included by loader_api.cpp, builder_api.cpp, query_api.cpp

extern "C" {
    // Plugin query function - returns pointer to API struct
    __attribute__((visibility("default"))) const void* plugin_query();
}
