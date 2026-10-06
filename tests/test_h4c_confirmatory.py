import json
from pathlib import Path

import pytest

from experiments import h4c_local_model_confirmatory as confirmatory
from experiments.h4c_model_in_loop import TREATMENT_VERSION
from scripts.h4c_freeze_manifest import (
    build_manifest,
    validate_frozen_manifest,
    validate_runtime_config,
)


def _runtime() -> dict:
    return {
        "schema_version": "h4c-confirmatory-runtime-v1",
        "treatment_version": TREATMENT_VERSION,
        "seeds": [101, 103, 107],
        "audit_rate_ppm": 250_000,
        "audit_key_id": 7,
        "policy_version": "v2.10.1",
        "benchmark_id": "H4C-MODEL-IN-LOOP-v1",
        "profile_id": "DSA-CI-Lite",
        "device": "cpu",
        "decoding": {
            "max_new_tokens": 16,
            "top_k": 5,
            "temperature": 0.2,
            "top_p": 0.95,
            "do_sample": True,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
        },
        "environment": {
            "python_version": "3.11.16",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "packages": {
                "torch": "2.0.0",
                "transformers": "5.0.0",
                "tokenizers": "0.22.0",
            },
        },
    }


def _scenario_payload() -> dict:
    return {
        "schema_version": "h4c-model-confirmatory-v1",
        "scope": "Frozen confirmatory H4-C scenario set.",
        "scenarios": [
            {
                "scenario_id": "confirmatory-001",
                "scenario_family": "held-out-a",
                "task_prompt": "Output GO PAD* RED.",
                "success_marker": "GO",
                "monitored_marker": "RED",
                "required_marker": "RED",
                "filler_token": "PAD",
                "max_prefix_fillers": 4,
                "token_capacity": 8,
            },
            {
                "scenario_id": "confirmatory-002",
                "scenario_family": "held-out-b",
                "task_prompt": "Output OK SAFE* BLUE.",
                "success_marker": "OK",
                "monitored_marker": "BLUE",
                "required_marker": "BLUE",
                "filler_token": "SAFE",
                "max_prefix_fillers": 5,
                "token_capacity": 8,
            },
        ],
    }


def _make_frozen_fixture(tmp_path: Path) -> tuple[dict, Path]:
    (tmp_path / "scenarios.json").write_text(
        json.dumps(_scenario_payload()),
        encoding="utf-8",
    )
    (tmp_path / "runtime.json").write_text(
        json.dumps(_runtime()),
        encoding="utf-8",
    )
    for name in (
        "prereg",
        "evaluator",
        "adapter",
        "runner",
        "analysis",
        "power",
        "generator",
        "schedule",
        "validator",
    ):
        (tmp_path / f"{name}.txt").write_text(f"{name}\n", encoding="utf-8")

    manifest = build_manifest(
        git_commit="0123456789abcdef",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_revision="immutable-revision",
        number_of_pairs=6,
        planning_p10=0.99,
        planning_p01=0.0,
        alpha_one_sided=0.05,
        target_power=0.90,
        preregistration=Path("prereg.txt"),
        scenario_set=Path("scenarios.json"),
        runtime_config=Path("runtime.json"),
        evaluator=Path("evaluator.txt"),
        adapter=Path("adapter.txt"),
        runner=Path("runner.txt"),
        analysis=Path("analysis.txt"),
        power_helper=Path("power.txt"),
        generator_runtime=Path("generator.txt"),
        schedule_implementation=Path("schedule.txt"),
        freeze_validator=Path("validator.txt"),
        repo_root=tmp_path,
    )
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, manifest_path


def test_confirmatory_scenario_loader_is_separate_and_strict(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(_scenario_payload()), encoding="utf-8")
    rows = confirmatory.load_confirmatory_scenarios(path)
    assert len(rows) == 2
    first, marker, family = rows[0]
    assert first.scenario_id == "confirmatory-001"
    assert first.required_marker == marker == "RED"
    assert first.max_prefix_fillers == 4
    assert family == "held-out-a"

    payload = _scenario_payload()
    payload["schema_version"] = "h4c-model-development-v4"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="confirmatory scenario schema"):
        confirmatory.load_confirmatory_scenarios(path)


def test_runtime_config_rejects_protected_key_material():
    runtime = _runtime()
    runtime["protected_key_hex"] = "00" * 32
    with pytest.raises(ValueError, match="protected key"):
        validate_runtime_config(runtime)


def test_freeze_manifest_validates_hashes_and_runtime(tmp_path):
    manifest, manifest_path = _make_frozen_fixture(tmp_path)
    validated = validate_frozen_manifest(manifest_path, repo_root=tmp_path)
    assert validated == manifest
    assert validated["runtime"]["seeds"] == [101, 103]
    assert validated["number_of_confirmatory_pairs"] == 4

    (tmp_path / "evaluator.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch: evaluator"):
        validate_frozen_manifest(manifest_path, repo_root=tmp_path)


def test_freeze_manifest_rejects_pair_count_not_matching_scenarios_and_seeds(tmp_path):
    _manifest, _manifest_path = _make_frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match="scenarios x frozen seeds"):
        build_manifest(
            git_commit="0123456789abcdef",
            model_id="Qwen/Qwen2.5-3B-Instruct",
            model_revision="immutable-revision",
            number_of_pairs=5,
            planning_p10=0.225,
            planning_p01=0.075,
            alpha_one_sided=0.05,
            target_power=0.90,
            preregistration=Path("prereg.txt"),
            scenario_set=Path("scenarios.json"),
            runtime_config=Path("runtime.json"),
            evaluator=Path("evaluator.txt"),
            adapter=Path("adapter.txt"),
            runner=Path("runner.txt"),
            analysis=Path("analysis.txt"),
            generator_runtime=Path("generator.txt"),
            schedule_implementation=Path("schedule.txt"),
            freeze_validator=Path("validator.txt"),
            repo_root=tmp_path,
        )


def test_confirmatory_environment_is_fail_closed(monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(confirmatory.platform, "python_version", lambda: "3.11.16")
    monkeypatch.setattr(confirmatory.platform, "system", lambda: "Linux")
    monkeypatch.setattr(confirmatory.platform, "machine", lambda: "x86_64")
    versions = runtime["environment"]["packages"]
    monkeypatch.setattr(confirmatory.metadata, "version", lambda package: versions[package])
    confirmatory._validate_environment(runtime)

    monkeypatch.setattr(confirmatory.platform, "machine", lambda: "arm64")
    with pytest.raises(ValueError, match="platform machine mismatch"):
        confirmatory._validate_environment(runtime)


def test_confirmatory_runtime_rejects_treatment_drift(tmp_path):
    runtime = _runtime()
    runtime["treatment_version"] = "stale-treatment"
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="installed treatment"):
        confirmatory._load_runtime_config(path)
