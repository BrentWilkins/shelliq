# Documentation compiler input-complete benchmark v4

Status: preregistered final generic safety correction.

V4 retains the unchanged v3 benchmark, index, compiler, provenance categories,
abstention contract, and 80%-coverage/95%-precision gate. Candidate ordering
changes only after requested-option completeness, request-literal completeness,
and source-count preference: instruction-intent similarity now precedes
unsupported-operand count. The compiler should select the semantically matching
recipe and then abstain on its unresolved operands, rather than deliver a simpler
wrong recipe.

No command-specific or reference-derived feature is allowed.
