# Documentation clarification continuation v2 — development results

Status: rejected on open development; sealed partition not opened.

After isolating fixture `MANPATH` and indexing the union of documented
command-family options, 15/16 flows completed exactly at 100% ready precision.
The remaining `nslookup` case uses the documented multi-character single-dash
option `-type=AXFR`. The verifier recognizes exact multi-character single-dash
options and inline `--long=value`, but only splits `=value` for double-dash long
options. It therefore interprets `-type=AXFR` as a short-option bundle and fails
closed despite the locally indexed `-type` option.

- safe initial abstentions: 16/16;
- stable, source-aligned intermediate flows: 16/16;
- invalid continuations failed closed: 16/16;
- exact completed flows: 15/16 (93.75%; required at least 95%);
- ready precision: 100%;
- semantic and local validity: 15/15 emitted commands.

No runtime or dataset changes were made and the frozen v2 sealed partition
remains unopened. A future runtime revision should generalize inline argument
handling to an exact indexed multi-character single-dash option, cover it with a
verifier unit test, and use another fresh command-disjoint evaluation.
