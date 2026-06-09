#include "query_q1.hpp"

#include "query_impl.hpp"

// Keep Q1 algorithm logic out of the template; base generation must write it.
void execute_q1(Engine&, const Q1Args&) {
    raise_missing_template_query_body("Q1");
}
