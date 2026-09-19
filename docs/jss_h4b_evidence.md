# H4-B Schedule-Aware Evasion Evidence

## Status

H4-B is complete and CI-validated on PR #40.

- Validated head: `388882200b551746895f06ce4a186f86879235f6`
- Merge commit on `main`: `7fe9bf9dad02c854e1ecd2edc8e2f1e0f4c27e62`
- Repository CI: success
- H4-B workflow: success
- Regression tests: 3 passed
- Benchmark design: 100 independently derived audit keys x 100 scenarios per key = 10,000 runs per event length
- Uploaded result artifact SHA-256: `784dec09740bf19eba36111fcfe5b7efb993ea32d9a8b24cca8280bfe09af658`

## Preregistered question

Can an attacker that knows a predictable public sampling schedule place concealment outside sampled positions, and does that avoidance transfer to a protected keyed schedule when the attacker knows the public algorithm, rate, and replay context but not the audit key?

## Results

| Event length | Analytical touch probability | Protected keyed exposure | Predictable public exposure |
| ---: | ---: | ---: | ---: |
| 1 | 0.00500 | 0.00494 | 0.00000 |
| 10 | 0.04889 | 0.04918 | 0.00000 |
| 50 | 0.22169 | 0.22256 | 0.00000 |
| 100 | 0.39423 | 0.39312 | 0.00051 |
| 500 | 0.91843 | 0.91880 | 0.91895 |

The public schedule is highly exploitable when the attacker has positional freedom. The same placement strategy does not transfer to the protected keyed schedule: protected exposure remains near the key-blind analytical expectation.

## Boundary condition

The 500-token event fills its complete 500-token placement bucket. The attacker therefore has no positional freedom, so public and protected exposure converge. This is a benchmark geometry boundary, not evidence that key protection stops mattering generally.

## Claim status

Supported by H4-B:
- protected keyed sampling resists schedule-specific placement avoidance in this synthetic benchmark;
- observed protected exposure remains near the analytical key-blind expectation for the tested event lengths.

Not established by H4-B:
- semantic deceptive-alignment detection;
- semantic validity of logits or probes;
- robustness to a model that adapts its internal representations in response to monitoring;
- general adversarial robustness outside the benchmark's placement model.

## Next empirical step

Move from synthetic placement-only evasion to a model-in-the-loop red-team study that can adapt token choice or internal representation under monitoring. Treat that study as a separate hypothesis and preregistration rather than as an extension of the current H4-B claim.
