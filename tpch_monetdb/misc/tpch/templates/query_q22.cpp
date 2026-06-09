#include "query_q22.hpp"

#include "query_impl.hpp"

// Keep Q22 algorithm logic out of the template; base generation must write it.
void execute_q22(Engine&, const Q22Args&) {
    raise_missing_template_query_body("Q22");
}
