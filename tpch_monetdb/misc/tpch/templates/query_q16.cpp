#include "query_q16.hpp"

#include "query_impl.hpp"

// Keep Q16 algorithm logic out of the template; base generation must write it.
void execute_q16(Engine&, const Q16Args&) {
    raise_missing_template_query_body("Q16");
}
