# Repository baseline before v2.10 alignment work

## Implemented version

The implementation is a Python package named `dualstream` at package version `0.1.0`. The compact evidence codec uses magic `DSAEV29` and supports legacy `0x0301` plus current compact evidence version `0x0302`; metadata strings identify `dsa-r-v2.9-signals`. No authoritative v2.10 manuscript, schema, or changelog was present in the repository or supplied in the session, so the starting implementation is best described as v2.9-compatible compact-evidence infrastructure, not v2.10-conformant.

## Architecture

* Language/runtime: Python 3.9+.
* Package management: `pyproject.toml` with setuptools and runtime dependencies on Torch, Transformers, NumPy, tqdm, FastAPI, and uvicorn; `requirements.txt` mirrors install dependencies for simple environments.
* Evidence schemas: binary frame codec in `dualstream/frame.py`, ARC sidecar frame types in `dualstream/arc_frame.py`, compact evidence in `dualstream/compact_evidence.py`, and vocabulary/event constants in `dualstream/vocab.py`.
* Encoder/decoder: compact sequence encoding and decoding are implemented in `dualstream/compact_evidence.py` with chunk CRCs, token records, selected-token binding by chosen rank or fallback chosen ID, sparse span records, metadata binding fields, and reconstruction helpers.
* Verifier: `dualstream/verifier.py` verifies compact artifacts, metadata binding, profile compatibility, retention floors, byte budgets, resource budgets, contiguous token indexes, score ranges, profile identifiers, and fail-closed structural errors.
* CLI: `dualstream/cli.py` exposes generation, evidence-budget verification, ARC solving, and Kaggle submission commands.
* CI: no workflow was present before this task.
* Tests: pytest tests cover frame codecs, compact evidence, verifier budgets, retention floor behavior, adaptive K, fallback behavior, API/service behavior, ARC solver artifacts, CLI paths, and no-pseudocode-term checks.
* Model integration: `dualstream/generator.py` integrates Hugging Face models and tokenizers, with optional offline cache behavior and generation configuration.
* Token hooks: generation captures per-token answer IDs, top-K evidence, optional attention, optional probes, randomized audit controls, fallback behavior, and compact evidence bytes.
* Probe/AST integration: probe-pack loading and randomized audit scheduling exist; AST vocabulary constants exist for verifier/resource/fallback events. Full activation-hook research infrastructure is not available.
* Existing benchmarks/results: ARC benchmark configuration and examples exist under `eval/` and `examples/`; there was no versioned empirical protocol, run manifest system, immutable raw artifact convention, or generated experimental reports.
* Random seeds/config/environment: generation accepts seeds; model/offline/cache/device configuration is recorded in generation metadata. Full run manifests with git, dependency, hardware, tokenizer, and artifact hashes were not previously implemented.

## Working commands observed

* `python -m pytest tests/test_experiment_framework.py -q`
* `python -m dualstream.cli verify-evidence-budget --artifact <path> --profile DSA-CI-Lite --ci-mode pr`
* `python -m dualstream.cli generate --prompt <text> --outdir <dir> --compact-evidence`

## Known incomplete features

* Missing authoritative v2.10 specification prevents full v2.10 conformance declaration.
* Evidence profile constants are implementation-carried v2.9-compatible values rather than confirmed v2.10 normative values.
* Signal-validity datasets, hidden-behavior labels, attacker checkpoints, and activation hooks are not bundled.
* Model-dependent experiments require external Hugging Face assets and may be limited by local hardware or network/cache state.

## Existing experimental capabilities

Before this task, the repository could generate compact evidence for model outputs and verify artifacts, but it did not provide a versioned experiment registry, protocol validation, deterministic run IDs, immutable raw artifacts, artifact hashing, reproducible synthetic systems conformance runs, or generated reports.

## Compatibility risks and files requiring modification

* `dualstream/compact_evidence.py`: future v2.10 schema magic, metadata keys, hash semantics, chunk layout, and token cardinality requirements may require changes.
* `dualstream/evidence_profile.py`: future v2.10 profile constants should be loaded from authoritative versioned configuration.
* `dualstream/verifier.py`: future v2.10 latency/memory budgets, assurance classes, DSA-R/DSA-P attestation rules, and fail-closed states may require stricter enforcement.
* `dualstream/cli.py`: experiment commands and protocol-validation commands require integration with the existing CLI.
* `tests/`: conformance coverage must expand once authoritative v2.10 requirements are available.
