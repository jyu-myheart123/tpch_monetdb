#include "query_q8.hpp"

#include "query_impl.hpp"

// Keep Q8 algorithm logic out of the template; base generation must write it.
void execute_q8(Engine&, const Q8Args&) {
    raise_missing_template_query_body("Q8");
}
