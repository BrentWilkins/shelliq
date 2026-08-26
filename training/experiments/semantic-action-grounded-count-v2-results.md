# Semantic-action grounded count v2 results

Status: completed; missed the five-acceptance gate.

The soft grounding prior restored the eight-record gate, reaching 8/8 exact and accepted at step 200. On the 92-record inner evaluation, validation loss selected epoch 5 at 4.044117. The selected checkpoint produced 1 exact and accepted action, 92 Rust-valid actions, and 79 first-command matches. Validity and command floors passed, but the primary 5/92 gate did not.

Attribution isolates the remaining requirement. Count top-1 is only 9/93, but top-eight is 87/93. With gold words and learned count search, 5 of the 23 fully covered records are exact and accepted; with both gold words and counts, all 23 are exact. With gold counts but learned words, only 2/92 are accepted. Global-only candidates remain 0/47 top-1. Post-command count prediction therefore made the requested five-acceptance ceiling reachable when words are correct, while semantic word retrieval remains dominant.

A training-prompt TF-IDF diagnostic improves global-only top-eight retrieval from 3/47 to 13/47 but reaches only 1/47 top-1. It is useful evidence but not sufficient as a standalone retriever. Outer validation and test remain sealed.
