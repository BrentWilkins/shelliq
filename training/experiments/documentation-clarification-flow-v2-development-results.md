# Documentation clarification flow v2 — development results

Status: rejected on open development; sealed partition not opened.

The unchanged runtime passed every safety and precision check on 16 fresh
commands, but exact placeholder identity was 13/16 (81.25%, below 95%). In each
disagreement, generic relevance ranking selected a different documented example
for the same command than the benchmark source row. The resulting question was
valid for that selected recipe, but it could not match the source row's label.

- safe default abstentions: 16/16;
- commandless structured abstentions: 16/16;
- focused questions: 16/16;
- exact frozen placeholder labels: 13/16;
- answered exact commands: 13/16;
- answered ready precision: 100%;
- semantic and local validity: 13/13 emitted commands;
- mean latency: 15.67 ms;
- maximum latency: 16.29 ms.

No runtime or dataset changes were made after this result. The frozen v2 sealed
partition remains unopened. A future evaluation must adjudicate clarification
against the runtime-selected documentation source, rather than assuming the
benchmark's source row wins generic retrieval.
