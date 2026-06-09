#include "query_q5.hpp"

#include "query_impl.hpp"

// Keep Q5 algorithm logic out of the template; base generation must write it.
void execute_q5(Engine&, const Q5Args&) {
    raise_missing_template_query_body("Q5");
}
