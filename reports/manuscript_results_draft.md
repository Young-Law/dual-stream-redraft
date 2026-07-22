# Manuscript results draft

The implementation was evaluated with protocol `dsa-v2-10-partial-systems-protocol-2026-07-22` in run `run-bd5d583cfd385ce29cc4f394194423f9` on synthetic controlled token replay. The implementation commit recorded by the run was `78e45bd41209209875b1983c5249c86b65d5bdcd`.

The implementation satisfied the run-level artifact requirements for `exp01_conformance`: it emitted `run_manifest.json`, `metrics.json`, `events.jsonl`, `failures.jsonl`, `environment.json`, and `artifact_hashes.json`. On the declared reference runner, the `DSA-CI-Lite` condition produced 64 evidence records for 64 synthetic generated tokens. The verifier rejected deterministic malformed artifacts at a rate of 1.000000 for the corruption subset exercised by the run.

The experiment observed storage and verifier timing metrics in `metrics.json`. Confidence intervals are not available because the current repository contains one completed systems-conformance run. The evidence did not demonstrate signal validity, deception detection, inner-alignment detection, or hidden-objective detection.
