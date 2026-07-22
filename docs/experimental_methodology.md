# Experimental methodology

The repository now separates implementation-validity experiments from signal-validity experiments. `exp01_conformance` is executable end to end with synthetic controlled replay. It measures evidence completeness, storage overhead, verifier reconstruction timing, memory observations, adaptive evidence behavior, and deterministic corruption rejection without requiring external models or datasets.

Signal-validity, ablation, monitor-evasion, and enforcement experiments are registered with explicit input requirements. They fail closed when required datasets, model checkpoints, label fields, activation hooks, or attacker checkpoints are absent.

Raw run artifacts are written under `results/raw/<run_id>/`. Derived summaries are regenerated under `results/summaries/`, tables under `results/tables/`, and future figures under `results/figures/`.
