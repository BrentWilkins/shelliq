# Salesforce CodeT5 production runtime evaluation v1 results

Status: failed preregistered gate; reject this checkpoint.

The frozen `Salesforce/codet5-small` semantic-action checkpoint ran through the
release-mode `shelliq suggest --json` path behind the new deterministic policy.
It passed 8/16 cases, entirely through abstention:

| Partition                 |      Gate |       Result | Outcome |
| ------------------------- | --------: | -----------: | ------- |
| Supported unseen commands |       3/4 |          0/4 | fail    |
| Unsupported requests      |       4/4 |          4/4 | pass    |
| Poisoned context          |       4/4 |          0/4 | fail    |
| Unsafe requests           |       4/4 |          4/4 | pass    |
| Warm CPU p95              | ≤5,000 ms | 10,161.39 ms | fail    |

## Findings

The model produced no correct ready suggestion for the four supported unseen
commands. Two outputs selected the literal executable name `Selection`, one
generation was rejected by the adapter, and one fell through to documentation
clarification. Rust grammar validity and familiar-command identity did not
generalize into usable semantic accuracy on new command families.

All poisoned-context requests abstained. Operand grounding explicitly rejected
invented model operands including `install`, `documentation`, and `sudo`. That
is safe behavior, but it does not satisfy the registered requirement to preserve
the requested command and operands. The boundary prevents bad output; it does
not make this model useful.

All unsafe requests abstained. Their commands were absent from the dedicated
index, so local command verification rejected them before the final unsafe-command
policy. Direct Rust tests separately cover that policy; this result is not an
isolated runtime test of its final layer.

The first evaluator process could not connect to loopback because its sandbox
denied sockets. It generated no adapter requests and is excluded. The unchanged
suite was then run once with explicit 127.0.0.1 permission.

## Decision

Do not ship or further tune this checkpoint. A new Salesforce experiment would
need a different task: rank complete documentation-derived recipes and abstain
when required slots are unresolved, instead of freely generating semantic
actions. It requires a fresh preregistration and command-disjoint suite.

## Frozen identities

- Preregistration commit:
  `1f7d33739478b575332b3515f22050f20e72bbcc`
- Suite SHA-256:
  `f3aac7d18c7dadd76ee6907976df666a961620aab41c6bb1c40eb24c4b73b58d`
- Checkpoint SHA-256:
  `f6248cf7aeb0bebb56853b60b4bb6633f7b665948fb549f6dcfc7336871957e7`
- ShellIQ binary SHA-256:
  `f58b5cc5f36dbf6889ebf1b1f82b270469d8bb6ac62813f4ce7eb4b081983bd6`
- Dedicated index SHA-256:
  `9f54a8b73d7f0122ebe4eeb22f43233c2e0eef57bece0949e5cc270f8e323759`
- Raw CPU report SHA-256:
  `1a1a7a145d9ea426e2ee8e2d1824fe54a91a1facabaf25211f029a160ca68cc9`
