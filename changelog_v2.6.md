# Changelog v2.6

## Scope
Operational revision to address critique on POC validity, CI practicality, red-state enforcement loops, and static-monitor evasion.

## Major additions
- Added v2.6 status note and abstract clarifying adversarial-operational scope and non-claims.
- Added distinction between host-induced divergence and model-internal divergence in Introduction.
- Added Section 2.1.3 with three POC threat levels (Level 0/1/2) and sufficiency guidance.
- Added computational budgeting assumption and static-probe non-goal.
- Added Section 4.2.2 for randomized/orthogonal probe selection with nonce-based schedule reconstruction requirements.
- Extended AST-1 with operational codes 304, 305, 530, 531.
- Added dynamic audit scheduling section with compact escalation algorithm.
- Added deterministic fallback routing semantics with bounded retries and fail-closed behavior.
- Reworked CI integration into tiers (0-3) and explicit smoke/targeted/nightly/release modes.
- Added Scenario C for internally motivated sycophancy/reward-hacking POC (design only).
- Added hypotheses H6-H9, benchmark families, and expanded metric set.
- Expanded limitations to cover proxy validity, false positives, scheduling misses, availability trade-offs, and randomized-probe assumptions.
- Added phased roadmap for v2.6 implementation.
- Added Appendices F-I with pseudocode for internally motivated POC, dynamic scheduler, fallback router, and nonce-randomized probe paths.

## Preserved core constraints
- DSA remains a technical proposal and evidence-capture contract, not a proof of inner alignment.
- DSA-R/DSA-P distinction retained.
- Monologue stream remains typed telemetry; chain-of-thought is not exposed to users.
- Minimum mandatory capture preserved: chosen token id + raw final-layer pre-control top-K logits.
- Probe outputs remain probabilistic and fallible.
