#include "query_q3.hpp"

#include "query_impl.hpp"

// Keep Q3 algorithm logic out of the template; base generation must write it.
void execute_q3(Engine&, const Q3Args&) {
    raise_missing_template_query_body("Q3");
}
