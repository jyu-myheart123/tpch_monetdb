#include "query_q11.hpp"

#include "query_impl.hpp"

// Keep Q11 algorithm logic out of the template; base generation must write it.
void execute_q11(Engine&, const Q11Args&) {
    raise_missing_template_query_body("Q11");
}
