# DSA v2.10 source behavior

This repository implements the DSA v2.10 software reference contract and its deterministic conformance suite. Empirical signal-validity and adversarial-evasion results are outside the scope of this implementation-alignment release.

The implemented source behavior is in `dualstream/v210.py`, with legacy dispatch integration in `dualstream/compact_evidence.py`, profile bindings in `dualstream/evidence_profile.py`, AST vocabulary in `dualstream/vocab.py`, and the CLI conformance command in `dualstream/cli.py`.

## DSA-R and DSA-P boundary

The software reference supports DSA-R local verification, hybrid selection replay, retention requirement creation, receipt validation, and a filesystem-backed reference validator. It does not claim genuine DSA-P device attestation. Real DSA-P assurance requires device-bound capture evidence, protected audit-key custody, signed governance artifacts, an authorized independent storage validator, valid unexpired receipts, and ongoing possession or restore checks.

## V3.3 and legacy compatibility

V3.3 uses wire version `0x0303` and includes sequence headers, chunks, token evidence, sparse spans, manifests, work certificates, retention requirements, and receipts. Decoding for `0x0301` and `0x0302` remains isolated; legacy artifacts remain legacy and are not silently rewritten.

## Local commands

- `python -m dualstream.cli conformance v2.10`
- `python -m pip install -e .`
- `python -m pytest`

Generated evidence artifacts are created in temporary directories during tests and conformance runs and are not committed.
