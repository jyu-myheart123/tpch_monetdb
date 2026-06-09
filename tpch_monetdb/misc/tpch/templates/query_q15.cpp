#include "query_q15.hpp"

#include "query_impl.hpp"

// Keep Q15 algorithm logic out of the template; base generation must write it.
void execute_q15(Engine&, const Q15Args&) {
    raise_missing_template_query_body("Q15");
}
