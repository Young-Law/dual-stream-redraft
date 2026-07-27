# Repository baseline before v2.10 alignment

Baseline inspection was performed on branch `work` before creating `codex/dsa-v210-clean-alignment`. The repository was a Python package named `dualstream` with a flat layout, examples, eval configuration, tests, and documentation directories.

## Architecture inspected

- `dualstream/compact_evidence.py` implemented the compact evidence encoder/decoder and token reconstruction.
- `dualstream/verifier.py` implemented compact artifact verification, byte budgets, local retention floor checks, and legacy resource-budget reporting.
- `dualstream/retention.py` computed compact evidence byte summaries and the v2.9 local reconstructability floor.
- `dualstream/evidence_profile.py` defined CI-Lite, CI-Standard, Deep, and Forensic byte ceilings and verifier budgets.
- `dualstream/cli.py` exposed generation, ARC solver, Kaggle export, and evidence-budget verification commands.
- `tests/` contained deterministic unit tests for the existing v2.6/v2.9 pipeline, compact evidence, profiles, budgets, retention floor, API, and ARC utilities.
- `pyproject.toml` lacked explicit setuptools package discovery for the flat repository layout.

## Wire formats implemented at baseline

- `0x0301`: legacy compact evidence layout with one-byte metadata length and chosen id stored per token.
- `0x0302`: current compact evidence layout with two-byte metadata length and chosen-rank plus fallback chosen-id semantics.

The baseline decoder dispatched on the encoded version for `0x0301` and `0x0302` and rejected unknown versions.

## Baseline test status

`python -m pytest` was run before implementation changes. Result: 98 passed, 1 warning in 43.96 seconds. The warning came from FastAPI/Starlette test-client compatibility.

## Identified v2.10 gaps

- No V3.3 wire version (`0x0303`) or V3.3 header/chunk/token/span/manifest structures.
- No portable deterministic verifier work certificate.
- Existing elapsed-time and memory checks still used v2.9-style resource-budget failure behavior.
- No explicit `INCONCLUSIVE_INFRA` retry semantics.
- Adaptive evidence was rank-oriented and did not implement keyed sampling, signed tension-map accounting, or canary separation.
- Retention was limited to a local byte floor and lacked signed retention requirements, receipts, lifecycle states, storage validation, possession challenge, migration, and restore workflows.
- AST-1 lacked v2.10 codes 523 through 528 and used 305 for the earlier randomized-audit label.
- CI did not explicitly run v2.10 conformance commands.
- Documentation did not distinguish the DSA-R software reference contract from real DSA-P device-bound production assurance.

## Compatibility and migration risks

- Existing public APIs in `dualstream.compact_evidence`, `dualstream.verifier`, `dualstream.retention`, and `dualstream.cli` must remain import-compatible for current tests and downstream users.
- V3.1 and V3.2 artifacts must remain legacy artifacts; signed or hashed legacy bytes cannot be silently rewritten as V3.3.
- V3.3 fields such as audit commitments, tension-map hashes, verifier-work certificates, retention requirements, and receipts cannot be reconstructed from legacy artifacts.

## Files expected to change

Implementation: compact evidence dispatch, evidence profiles, AST vocabulary, CLI, new v2.10 reference module. Tests: deterministic v2.10 conformance tests. Configuration: package discovery, ignore rules, CI workflow. Documentation: baseline, conformance matrix, implementation, migration, portable verifier, hybrid evidence, retention assurance, and security model.
