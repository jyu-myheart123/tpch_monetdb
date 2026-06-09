#include "query_q7.hpp"

#include "query_impl.hpp"

// Keep Q7 algorithm logic out of the template; base generation must write it.
void execute_q7(Engine&, const Q7Args&) {
    raise_missing_template_query_body("Q7");
}
