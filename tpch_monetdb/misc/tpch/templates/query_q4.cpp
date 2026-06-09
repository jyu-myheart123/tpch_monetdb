#include "query_q4.hpp"

#include "query_impl.hpp"

// Keep Q4 algorithm logic out of the template; base generation must write it.
void execute_q4(Engine&, const Q4Args&) {
    raise_missing_template_query_body("Q4");
}
