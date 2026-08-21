# Local teacher selection v2

All full runs used the audited 20-case `teacher-selection-v2.jsonl` set,
temperature zero, no hidden reasoning, and server-side JSON-object constrained
decoding. Exact-document scores are intentionally strict; split versus clustered
short flags and harmless lexical quoting can be behaviorally equivalent.

| model | first command | exact flags | exact document | latency | residency |
|---|---:|---:|---:|---:|---:|
| Qwen 3.8 27B 80K | 100% | **90%** | **60%** | 28.5 s | 17 GB, 100% GPU |
| Qwen 3.6 27B | 100% | 70% | 55% | 45.8 s | 19 GB, 100% GPU |
| GLM-4.7 Flash Coder | 95% | 60% | 45% | **27.1 s** | 21 GB, 100% GPU |
| Qwen3-Coder 30B-A3B | 100% | 70% | 45% | 32.9 s | 24 GB, 92% GPU |

GPT-OSS 20B was not promoted to a full run. Ollama did not emit final JSON with
`reasoning_effort=none`; its supported `low` mode produced 37.5% exact flags and
documents on the first eight cases.

## Decision

Use `qwen3.8:27b-80k` for bulk local proposals. It leads both behavioral metrics,
fits fully in VRAM, and is fast enough that further local screening has little
expected value. It is not an authority: proposals still require syntax/semantic
verification and human review, with a stronger hosted teacher reserved for hard
disagreements.

### Reasoning-effort sweep

| effort | exact flags | exact document | JSON parse | latency |
|---|---:|---:|---:|---:|
| none | **90%** | 60% | 100% | **28.5 s** |
| low | 80% | 60% | 100% | 68.4 s |
| medium | 80% | **65%** | 100% | 67.4 s |
| high | 75% | **65%** | 95% | 111.6 s |

Medium reasoning corrected genuinely wrong `brew` and `jq` outputs from the fast
pass, but also dropped a required `grep -n`. High added no document accuracy,
was nearly four times slower than none, and emitted one malformed response.
Therefore use a verifier-directed cascade: generate at `none`, retry only rejected
records at `medium`, then escalate unresolved disagreements. Do not globally use
medium or high.

The disagreement review also identified verifier work. `git clean -n -d -x` and
`git clean -ndx`, or `df -i -h` and `df -ih`, should compare as equivalent option
sequences. Quote-only differences around a regex word can also preserve the same
rendered shell command. These normalizations must be explicit and tested; never
repair malformed teacher JSON after generation.
