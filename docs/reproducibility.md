# Reproducibility

Validate the frozen protocol with:

```bash
python -m dualstream.cli experiment validate-protocol --protocol experiments/protocols/dsa_v2_10_protocol.yaml
```

Run the systems conformance experiment with:

```bash
python -m dualstream.cli experiment run --protocol experiments/protocols/dsa_v2_10_protocol.yaml --experiment exp01_conformance
```

Verify hashes for a completed run with:

```bash
python -m dualstream.cli experiment verify-hashes --run-dir results/raw/<run_id>
```

Regenerate summaries and tables with:

```bash
python -m dualstream.cli experiment summarize
python -m dualstream.cli experiment tables
```

Every run manifest records the run ID, experiment ID, protocol ID, git commit, working-tree status, implementation version, schema ID/hash, model/tokenizer/probe/benchmark identifiers, split, seed, hardware, OS, runtime, dependency versions, decoding configuration, evidence profile, verifier budget, retention policy, command, and exit status.
