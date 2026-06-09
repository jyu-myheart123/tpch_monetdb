#include "query_q14.hpp"

#include "query_impl.hpp"

// Keep Q14 algorithm logic out of the template; base generation must write it.
void execute_q14(Engine&, const Q14Args&) {
    raise_missing_template_query_body("Q14");
}
