# DSA v2.9 changelog

This release adds DSA-R v2.9 compact evidence reference behavior while keeping the v2.6 `MonologueFrameV1` path available.

## Added

- Profile-aware compact evidence artifacts for CI-Lite, CI-Standard, Deep, and Forensic review modes.
- Rank-triggered adaptive evidence capture keyed to chosen-token rank rather than entropy.
- Compact artifact verification with structural, integrity, profile, retention-floor, and budget checks.
- Mechanical retention-floor helpers that reject summary-only or truncated artifacts.
- CLI verification through `python -m dualstream.cli verify-evidence-budget`.

## Scope

This repository remains a software-only DSA-R reference implementation. DSA-P device attestation is outside the current code path.

## v2.9 conformance hardening

- CI-Lite now defaults to base K=3 with chosen-rank-triggered adaptive widening through rank 10.
- Compact evidence metadata records the explicit v3.1 DSA-R profile, signal schema, probe pack, decoder-control, adaptive policy, verifier budget, retention-floor policy, and quantization identifiers.
- The verifier reports short-fixture budget status separately from canonical 10,000-token profile conformance and supports `--strict-profile-budget` plus `--enforce-rss-budget`.
- Local retention-floor accounting includes the compact header, profile metadata/manifest bytes, chunk headers, token records, fallback chosen-id bytes, and sparse span events.
- AST 521 reports retention-floor violations; AST 522 reports verifier resource-budget violations.
- DSA-Forensic remains opt-in for incident/forensic modes and is not accepted as a PR/normal CI default.
- The implementation remains DSA-R reference behavior and does not implement DSA-P device/enclave attestation.
