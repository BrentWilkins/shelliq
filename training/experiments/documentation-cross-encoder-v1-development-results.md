# Documentation-conditioned cross-encoder v1 development results

Status: development gate passed; sealed test authorized but not yet scored.

The frozen CodeT5 encoder plus learned scalar head trained on 7,356 pairs across
512 commands. Epoch 48 was selected by the preregistered development rule.

| Development gate                      |        Required |        Result |
| ------------------------------------- | --------------: | ------------: |
| Sufficient unseen-command ready       | ≥45/64 and ≥70% | 58/64 (90.6%) |
| Insufficient-documentation abstention |           64/64 |         64/64 |
| Development pair truncation           |        reported |             0 |

The selected abstention threshold is `-0.31467324495315546`. One training pair
was truncated at the frozen 256-token input limit; no development pair was
truncated. Encoder caching took 87.72 seconds on CPU.

Every ready development result selected the labeled documentation record,
resolved all original generic slots from request-visible values, and passed the
Rust semantic-action encode/decode check. Development commands are disjoint from
both supervised training and the sealed 128-command test split.

## Frozen identities before test

- Manifest SHA-256:
  `152ba19746d649fff2ee99342cd25f839e2646041003354667859e1330ccc356`
- Training cases SHA-256:
  `b027a0182b1222534f79287306e27a62d5eeb144158bf9a95809c018a20260c6`
- Development cases SHA-256:
  `307041ef5fc5467aa8315eb5d8ae9a13e106ff797db278a5f13eb7edf886cfd7`
- Sealed test cases SHA-256:
  `d0d9e6c8ce97bcdbbd4834e44ee43f56294cc77d181152446f3dc8753e42ba2f`
- Selected checkpoint SHA-256:
  `138b6c79ed36458308fff038672e195a6ccc54c3fd4f0c584200b3e533f15cd4`
- Raw development report SHA-256:
  `8cbbee485679419a607243115be3a4c02df5e8f1fb15cf5a6bc1d12054dfceec`
