# Reviewer Response Memo (v2.6)

This revision directly addresses the critique in four concrete ways while preserving the paper's original caution level.

## 1) "Forced compliance is not internally motivated deception"
You were correct that host-forced behavior only validates plumbing. We now explicitly separate POC levels:
- Level 0 = host-induced divergence (plumbing only),
- Level 1 = internally motivated proxy (reward-hacking/sycophancy),
- Level 2 = adaptive monitor-aware evasion.

We also added Scenario C and Appendix F to define a Level-1 design where agreement with false premises is reward-incentivized, so answer/evidence divergence can emerge without host-forced decoding.

## 2) "CI auditing may be too expensive"
We replaced implicit exhaustive auditing with dynamic scheduling and tiering. Cheap checks run broadly; expensive probes run only on suspicious windows or high-risk suites. We explicitly separate commit-blocking smoke tests from nightly/release deep audits and require runtime/storage reporting.

## 3) "Red-state enforcement may loop"
We added deterministic fallback routing: bounded retry budgets, generation halt, cache isolation/flush options, safe-route injection, and fail-closed behavior when budgets are exhausted. This is meant to prevent infinite same-context retries and reduce DoS-style amplification.

## 4) "Static probes may be learnably evaded"
We added randomized/orthogonal audit paths (nonce-conditioned sampling, rotating probe subsets, verifier-reconstructable schedules) and dedicated metrics to compare static vs randomized evasion success and cost. We also state clearly that this increases attack cost but does not prove impossibility.

## What remains intentionally unchanged
- The paper still does not claim proof of inner alignment.
- The monologue remains typed telemetry, not user-exposed chain-of-thought.
- Probe outputs remain probabilistic.
- DSA-P attestation assumptions remain explicit.

In short: v2.6 narrows overclaim risk, improves operational realism, and makes the next empirical stage more scientifically diagnostic.
