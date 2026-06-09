#include "query_q12.hpp"

#include "query_impl.hpp"

// Keep Q12 algorithm logic out of the template; base generation must write it.
void execute_q12(Engine&, const Q12Args&) {
    raise_missing_template_query_body("Q12");
}
