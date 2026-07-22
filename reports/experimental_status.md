# Experimental status

## Tested version

* DSA specification status tested: 0.1.0-v2.10-partial with protocol `dsa-v2-10-partial-systems-protocol-2026-07-22`.
* Implementation commit recorded by the run: `78e45bd41209209875b1983c5249c86b65d5bdcd`.
* Run ID: `run-bd5d583cfd385ce29cc4f394194423f9`.

## Completed experiments

* `exp01_conformance` completed on synthetic controlled token replay with 64 tokens.

## Evaluated hypotheses

Implementation-validity hypotheses evaluated in this run: evidence completeness, reconstruction correctness, corruption rejection, storage overhead, verifier latency, verifier memory observation, retention-floor measurement, and adaptive-evidence behavior.

Signal-validity hypotheses were not evaluated. The run used synthetic token streams and does not test deception detection, inner alignment, hidden objectives, or behavioral monitoring utility.

## Actual measured results

For `DSA-CI-Lite`, the run observed 64 evidence records for 64 generated tokens, evidence completeness rate 1.000000, raw bytes/token 30.265625, compressed bytes/token 16.203125, verifier p50 0.004118189s, verifier p95 0.005750725s, verifier max 0.006671738s, adaptive record fraction 0.000000, and malformed-artifact rejection rate 1.000000.

Confidence intervals were not reported because only one completed run exists. The metrics file and artifact hashes preserve source-run mapping.

## Failed or excluded runs

No failed or excluded runs were recorded in the generated summary. Corruption fixtures are retained in `failures.jsonl` with expected rejection outcomes.

## Limitations

The authoritative v2.10 specification is missing. Results concern implementation validity only. Synthetic replay isolates evidence-format behavior from model sampling variance but cannot demonstrate signal validity.

## Claims supported

The completed run supports the limited claim that the implemented synthetic systems conformance runner produced hashed artifacts and measured verifier/storage behavior for existing compact evidence profiles on the declared runner.

## Claims not supported

The evidence does not demonstrate that DSA detects deception, proves alignment, reads hidden thoughts, or generalizes to behavioral benchmarks.
