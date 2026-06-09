#include "query_q10.hpp"

#include "query_impl.hpp"

// Keep Q10 algorithm logic out of the template; base generation must write it.
void execute_q10(Engine&, const Q10Args&) {
    raise_missing_template_query_body("Q10");
}
