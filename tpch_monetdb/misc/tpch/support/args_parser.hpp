#pragma once

#include <string>

struct QueryRequest {
    std::string id;
    std::string line;
};

// Supported query IDs for generated runtime fallback parsing.
inline bool is_supported_query_id(const std::string& qid) {
    return qid == "1" || qid == "2" || qid == "3" || qid == "4" ||
           qid == "5" || qid == "6" || qid == "7" || qid == "8" ||
           qid == "9" || qid == "10" || qid == "11" || qid == "12" ||
           qid == "13" || qid == "14" || qid == "15" || qid == "16" ||
           qid == "17" || qid == "18" || qid == "19" || qid == "20" ||
           qid == "21" || qid == "22";
}
