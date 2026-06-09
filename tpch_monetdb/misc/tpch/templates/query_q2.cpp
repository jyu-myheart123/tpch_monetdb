#include "query_q2.hpp"

#include "query_impl.hpp"

// Keep Q2 algorithm logic out of the template; base generation must write it.
void execute_q2(Engine&, const Q2Args&) {
    raise_missing_template_query_body("Q2");
}
