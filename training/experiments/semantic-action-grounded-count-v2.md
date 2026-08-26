# Semantic-action grounded count v2

Status: preregistered before implementation evaluation.

V1's hard prompt-grounding mask reached near-zero teacher-forced loss but only 1/8 exact overfit because required global-only candidates were unreachable. V2 replaces only that hard mask with an explicitly supervised soft grounding prior.

## Frozen changes

1. Retain v1's post-command argument-count alignment, spelled-number normalization, top-eight count expansion, width-32 beam, v3 candidate representations, split, optimizer, and validation protocol.
2. Remove hard candidate masking from training and inference. All role-eligible candidates remain reachable.
3. Add a two-class linear grounding head at every decoder position. The classes are `prompt_grounded` when the gold candidate is local or derived and `global_only` otherwise. Supervise it only at argument candidate positions with weight 0.10.
4. Add the selected grounding class log-probability to each argument candidate logit before both candidate cross-entropy and beam decoding. Command candidate logits and content-only auxiliary logits remain unchanged.
5. No fixed grounding bonus, per-record rule, new data, inventory expansion, target-derived feature, outer/test access, or post-evaluation tuning is allowed.

## Gates

- CPU: finite action, candidate, count, content-global, grounding, and total losses and gradients; bounded grammar generation.
- Eight-record overfit: 8/8 exact and accepted within 2,000 CUDA updates. Failure stops before inner evaluation.
- Inner: same 518/92 split and 50-epoch validation-loss selection.
- Floors: at least 90% Rust-valid and 80% first-command match.
- Success: at least 5/92 reference accepted.

Outer validation and test remain sealed.
