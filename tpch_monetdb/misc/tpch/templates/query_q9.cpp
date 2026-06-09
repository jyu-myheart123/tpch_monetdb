#include "query_q9.hpp"

#include "query_impl.hpp"

// Keep Q9 algorithm logic out of the template; base generation must write it.
void execute_q9(Engine&, const Q9Args&) {
    raise_missing_template_query_body("Q9");
}
