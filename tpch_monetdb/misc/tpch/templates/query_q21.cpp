#include "query_q21.hpp"

#include "query_impl.hpp"

// Keep Q21 algorithm logic out of the template; base generation must write it.
void execute_q21(Engine&, const Q21Args&) {
    raise_missing_template_query_body("Q21");
}
