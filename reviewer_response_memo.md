# Reviewer Response Memo (v2.10 / H4-B integration)

This revision retains the paper's original caution level while converting the hybrid-sampling claim from a proposal-only statement into a CI-backed synthetic mechanism result.

## 1) Predictable audit schedules can be evaded

The concern is supported by H4-B. A schedule-aware placement attacker with access to a predictable public schedule reduced public-schedule exposure to 0 for 1-, 10-, and 50-token events and to 0.00056 for 100-token events.

## 2) Protected keyed sampling prevents transfer of that schedule-specific avoidance

The corrected rerun holds the public algorithm, rate, scenario context, and `audit_key_id=7` constant across the paired schedules; only the secret key material differs. Without the protected audit key, the same placement strategy did not transfer to the keyed schedule. Protected exposure stayed near the key-blind analytical expectation:

- length 1: analytical 0.00500, protected 0.00517;
- length 10: analytical 0.04889, protected 0.04981;
- length 50: analytical 0.22169, protected 0.22336;
- length 100: analytical 0.39423, protected 0.39370;
- length 500: analytical 0.91843, protected 0.91905.

The 500-token case is a boundary condition because the event fills the entire placement bucket and leaves no avoidance freedom.

## 3) Reproducibility

H4-B was executed in GitHub Actions with:
- 100 independently derived audit keys;
- 100 scenarios per key;
- 10,000 runs per event length;
- a regression test that asserts identical public replay-context inputs for the paired schedules and distinct secret key material;
- successful H4-B workflow run #8;
- successful repository CI run #111.

Corrected validated head: `ea35fd4f04c762935366b815f21930d6e20a91ff`

Result JSON SHA-256: `b419fc2c833b407bdbe59f3780bcb89f364244e0903cd2045a6ee7206adbbc61`

GitHub Actions artifact digest (ZIP): `sha256:792670e21d25e970ba8f5345e0f4cf03a194096170bc8772001e83abb3be2f30`

## 4) Claim boundary remains narrow

The result validates a synthetic schedule-unpredictability mechanism under the benchmark's placement model. It does not establish semantic deceptive-alignment detection, semantic validity of logits or probes, or robustness to a model that adapts its internal representations.

## What remains future work

The next empirical stage is a preregistered model-in-the-loop adaptive-evasion study. It should test whether an actual model can adapt token choices or internal representations under monitoring and whether hybrid sampling provides incremental detection value in that setting.
