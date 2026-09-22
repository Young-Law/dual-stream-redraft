# H4-B Schedule-Aware Evasion Evidence

## Status

H4-B is complete and revalidated on PR #41 after correcting the paired replay-context control.

- Corrected validated head: `620101bb7c91b8447956d0a77ce9e4d8fb584499`
- Repository CI run #114: success
- H4-B workflow run #9: success
- Regression tests include an explicit paired replay-context invariant
- Benchmark design: 100 independently derived audit keys x 100 scenarios per key = 10,000 runs per event length
- Result JSON SHA-256: `b419fc2c833b407bdbe59f3780bcb89f364244e0903cd2045a6ee7206adbbc61`
- GitHub Actions artifact digest (ZIP): `sha256:75573fc39b40c9168b8882261b117a7ec4f9a69f1811d7d670cc1d37b80bb3e8`
- Replay-context control: both schedules use `audit_key_id=7`; only secret key material differs.

## Preregistered question

Can an attacker that knows a predictable public sampling schedule place concealment outside sampled positions, and does that avoidance transfer to a protected keyed schedule when the attacker knows the public algorithm, rate, and replay context but not the audit key?

## Results

| Event length | Analytical touch probability | Protected keyed exposure | Predictable public exposure |
| ---: | ---: | ---: | ---: |
| 1 | 0.00500 | 0.00517 | 0.00000 |
| 10 | 0.04889 | 0.04981 | 0.00000 |
| 50 | 0.22169 | 0.22336 | 0.00000 |
| 100 | 0.39423 | 0.39370 | 0.00056 |
| 500 | 0.91843 | 0.91905 | 0.91829 |

The public schedule is highly exploitable when the attacker has positional freedom. After holding the public replay context constant across the paired schedules, the same placement strategy does not transfer to the protected keyed schedule: protected exposure remains near the key-blind analytical expectation.

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
