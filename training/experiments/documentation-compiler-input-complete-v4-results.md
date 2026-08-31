# Documentation compiler input-complete v4 results

Status: passed with 98.90% ready precision.

V4 delivered 90 exact typed commands across 100 unseen command families,
withheld nine, and produced one false-ready result. All 91 ready documents passed
the Rust action round trip. Exact coverage remained 90%; ready precision improved
from 96.77% to 98.90%.

The sole false-ready request contained a literal already embedded in its exact
documentation instruction plus a conflicting appended literal. Literal
completeness preceded intent similarity and selected a different recipe that used
both. V5 may rank semantic intent before literal completeness, causing the exact
recipe to expose the unused conflicting literal and abstain.

Raw report SHA-256:
`012885d23f86d2ff20c29f3e6bc96cca05354c52990e084d12be6d003f16d1a6`.
