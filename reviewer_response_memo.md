# Reviewer Response Memo (v2.10 / H4-B integration)

This revision retains the paper's original caution level while converting the hybrid-sampling claim from a proposal-only statement into a CI-backed synthetic mechanism result.

## 1) Predictable audit schedules can be evaded

The concern is supported by H4-B. A schedule-aware placement attacker with access to a predictable public schedule reduced public-schedule exposure to 0 for 1-, 10-, and 50-token events and to 0.00051 for 100-token events.

## 2) Protected keyed sampling prevents transfer of that schedule-specific avoidance

With the same public algorithm, rate, and replay context but without the protected audit key, the same placement strategy did not transfer to the keyed schedule. Protected exposure stayed near the key-blind analytical expectation:

- length 1: analytical 0.00500, protected 0.00494;
- length 10: analytical 0.04889, protected 0.04918;
- length 50: analytical 0.22169, protected 0.22256;
- length 100: analytical 0.39423, protected 0.39312;
- length 500: analytical 0.91843, protected 0.91880.

The 500-token case is a boundary condition because the event fills the entire placement bucket and leaves no avoidance freedom.

## 3) Reproducibility

H4-B was executed in GitHub Actions with:
- 100 independently derived audit keys;
- 100 scenarios per key;
- 10,000 runs per event length;
- passing H4-B regression tests;
- successful repository CI.

Validated head: `388882200b551746895f06ce4a186f86879235f6`

Artifact SHA-256: `784dec09740bf19eba36111fcfe5b7efb993ea32d9a8b24cca8280bfe09af658`

## 4) Claim boundary remains narrow

The result validates a synthetic schedule-unpredictability mechanism under the benchmark's placement model. It does not establish semantic deceptive-alignment detection, semantic validity of logits or probes, or robustness to a model that adapts its internal representations.

## What remains future work

The next empirical stage is a preregistered model-in-the-loop adaptive-evasion study. It should test whether an actual model can adapt token choices or internal representations under monitoring and whether hybrid sampling provides incremental detection value in that setting.
