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
