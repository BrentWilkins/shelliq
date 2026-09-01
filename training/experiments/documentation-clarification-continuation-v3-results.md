# Documentation clarification continuation v3 results

Status: accepted.

The preregistered exact-indexed single-dash inline-argument change passed both
fresh command-disjoint partitions:

| Metric | Development | Sealed test |
| --- | ---: | ---: |
| Records | 16 | 48 |
| Safe initial abstentions | 16/16 | 48/48 |
| Source-aligned questions | 16/16 | 48/48 |
| Stable intermediate abstentions | 16/16 | 48/48 |
| Invalid continuations failed closed | 16/16 | 48/48 |
| Exact completed flows | 16/16 | 48/48 |
| Ready precision | 100% | 100% |
| Semantic and local validity | 100% | 100% |
| Mean latency | 16.19 ms | 14.75 ms |
| Maximum latency | 27.56 ms | 18.21 ms |

The sealed partition was opened once after the runtime, harness, dataset, and
hashes were frozen. Its completion rate was 100%, above the 95% gate.

Regression evaluation retained 100% ready precision and semantic/local
validity. Source-aligned v1 emitted 47/48 exact commands while safely abstaining
on the remaining case. Fallback v3 and v4 reproduced 42 and 35 exact ready
outputs respectively, and all 32 incomplete cases produced explicit commandless
abstentions. The fallback reports retain their historical `gate_passed: false`
coverage result; v3's 15/16 safe-incomplete metric also predates this experiment
and is unchanged by the verifier revision.
