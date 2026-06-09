#include "query_q6.hpp"

#include "query_impl.hpp"

// Keep Q6 algorithm logic out of the template; base generation must write it.
void execute_q6(Engine&, const Q6Args&) {
    raise_missing_template_query_body("Q6");
}
