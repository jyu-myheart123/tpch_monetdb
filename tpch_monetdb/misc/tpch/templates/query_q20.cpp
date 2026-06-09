#include "query_q20.hpp"

#include "query_impl.hpp"

// Keep Q20 algorithm logic out of the template; base generation must write it.
void execute_q20(Engine&, const Q20Args&) {
    raise_missing_template_query_body("Q20");
}
