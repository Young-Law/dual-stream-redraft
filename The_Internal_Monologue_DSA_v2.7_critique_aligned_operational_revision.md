# The Internal Monologue: DSA v2.7 Critique-Aligned Operational Revision

This v2.7 paper is the current revision. It preserves v2.6 adversarial-operational content and adds:

1. Coherence Audit mechanics before high-stakes scenarios.
2. CI/CD as the operational proving ground for theory.
3. Concrete telemetry-exhaust handling: ring buffers, pass-slice commitments, frozen elevated-risk windows, Tier 3/deep retention, and columnar heavy-telemetry storage.

## Safety boundaries

- DSA does not prove inner alignment.
- DSA does not prove deception-detection completeness.
- DSA-R is not tamper-resistant against a compromised host.
- Randomized telemetry is not a formal robustness proof.
- This reference implementation is not production-grade DSA-P.

## Operational framing

Coherence audits aggregate token evidence into span-level findings before scenario analysis. AST 101/230/305/530/531 are first-class operational signals.

CI modes (`smoke`, `targeted`, `nightly`, `release-blocking`, `deep`) instantiate the policy in operations. Tier 3 and deep CI modes retain full telemetry by default.
