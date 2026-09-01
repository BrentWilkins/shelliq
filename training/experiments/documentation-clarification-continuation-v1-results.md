# Documentation clarification continuation v1 — sealed results

Status: rejected by the preregistered sealed gate.

The frozen 48-case partition was opened once against runtime `2a50d93`.

- safe initial abstentions: 48/48;
- source-aligned initial clarifications: 47/48 (97.9%);
- stable intermediate flows: 47/48;
- invalid continuations failed closed: 48/48;
- exact completed flows: 44/48 (91.7%; required at least 95%);
- ready precision: 100%;
- semantic and local validity: 44/44 emitted commands;
- mean complete-flow latency: 17.57 ms;
- maximum complete-flow latency: 31.67 ms.

No post-sealed runtime tuning or case replacement was performed. Failure review
identified two harness limitations. The fake executable indexed options only
from the dataset source row, while runtime retrieval may select another recipe
for the same command; three otherwise complete commands were therefore rejected
by an incomplete fixture index. The alignment oracle also required a unique
command/intent row, but `more` has equivalent common and Linux page records with
the same intent and operand.

The runtime itself retained 100% ready precision. On the previously frozen
48-case source-aligned set, the new continuation flow completed 48/48 exactly,
including the three formerly incomplete answered cases. Frozen v3/v4 retained
100% ready precision and all 32 incomplete requests remained commandless, though
conservative ready coverage changed from 81 to 77 emitted commands.

A measurement-only v2 should keep the runtime frozen, build fixture option facts
from every documentation recipe for the command family, and accept duplicate
normalized intent rows when the reported operand exists in a matching recipe.
