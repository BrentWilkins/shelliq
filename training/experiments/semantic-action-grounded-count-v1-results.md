# Semantic-action grounded count v1 results

Status: failed the overfit gate; inner evaluation was not opened.

The CPU gate passed with finite losses and gradients. On the fixed eight-record CUDA sample, teacher-forced loss fell from 0.288150 at step 100 to 0.000781 at step 400, while exact decoding remained 1/8 at every measurement. The run was stopped because the hard inference mask excludes required global-only candidates: additional optimization cannot recover an output removed from the search space.

The post-command count alignment and deterministic number normalization remain reusable. The failed component is specifically the hard local/derived candidate mask. Its replacement must keep every candidate reachable and learn when prompt grounding should outweigh the global inventory.
