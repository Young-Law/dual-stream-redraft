import json
from pathlib import Path

import pytest

from experiments import h4c_local_model_confirmatory as confirmatory
from experiments.h4c_model_in_loop import TREATMENT_VERSION
from scripts import h4c_freeze_manifest as freeze_manifest
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
        "key_commitment": "a" * 64,
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


def _make_frozen_fixture(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    frozen_git_commit = "0" * 40
    monkeypatch.setattr(
        freeze_manifest,
        "_resolve_git_commit",
        lambda _repo_root=Path("."): frozen_git_commit,
    )

    (tmp_path / "scenarios.json").write_text(
        json.dumps(_scenario_payload()),
        encoding="utf-8",
    )
    (tmp_path / "runtime.json").write_text(
        json.dumps(_runtime()),
        encoding="utf-8",
    )

    canonical_files = (
        Path("docs/h4c_model_in_loop_preregistration.md"),
        Path("experiments/h4c_model_in_loop.py"),
        Path("experiments/h4c_local_hf_adapter.py"),
        Path("experiments/h4c_local_model_confirmatory.py"),
        Path("experiments/h4c_paired_analysis.py"),
        Path("experiments/h4c_power.py"),
        Path("dualstream/generator.py"),
        Path("dualstream/compact_evidence.py"),
        Path("scripts/h4c_freeze_manifest.py"),
    )
    for relative_path in canonical_files:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path.as_posix() + "\n", encoding="utf-8")

    manifest = build_manifest(
        git_commit=frozen_git_commit,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_revision="immutable-revision",
        number_of_pairs=6,
        planning_p10=0.99,
        planning_p01=0.0,
        alpha_one_sided=0.05,
        target_power=0.90,
        preregistration=Path("docs/h4c_model_in_loop_preregistration.md"),
        scenario_set=Path("scenarios.json"),
        runtime_config=Path("runtime.json"),
        evaluator=Path("experiments/h4c_model_in_loop.py"),
        adapter=Path("experiments/h4c_local_hf_adapter.py"),
        runner=Path("experiments/h4c_local_model_confirmatory.py"),
        analysis=Path("experiments/h4c_paired_analysis.py"),
        power_helper=Path("experiments/h4c_power.py"),
        generator_runtime=Path("dualstream/generator.py"),
        schedule_implementation=Path("dualstream/compact_evidence.py"),
        freeze_validator=Path("scripts/h4c_freeze_manifest.py"),
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


def test_freeze_manifest_validates_hashes_and_runtime(tmp_path, monkeypatch):
    manifest, manifest_path = _make_frozen_fixture(tmp_path, monkeypatch)
    validated = validate_frozen_manifest(manifest_path, repo_root=tmp_path)
    assert validated == manifest
    assert validated["runtime"]["seeds"] == [101, 103, 107]
    assert validated["number_of_confirmatory_pairs"] == 6

    (tmp_path / "experiments/h4c_model_in_loop.py").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch: evaluator"):
        validate_frozen_manifest(manifest_path, repo_root=tmp_path)


def test_freeze_manifest_rejects_pair_count_not_matching_scenarios_and_seeds(tmp_path, monkeypatch):
    _manifest, _manifest_path = _make_frozen_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="scenarios x frozen seeds"):
        build_manifest(
            git_commit="0" * 40,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            model_revision="immutable-revision",
            number_of_pairs=5,
            planning_p10=0.99,
            planning_p01=0.0,
            alpha_one_sided=0.05,
            target_power=0.90,
            preregistration=Path("docs/h4c_model_in_loop_preregistration.md"),
            scenario_set=Path("scenarios.json"),
            runtime_config=Path("runtime.json"),
            evaluator=Path("experiments/h4c_model_in_loop.py"),
            adapter=Path("experiments/h4c_local_hf_adapter.py"),
            runner=Path("experiments/h4c_local_model_confirmatory.py"),
            analysis=Path("experiments/h4c_paired_analysis.py"),
            power_helper=Path("experiments/h4c_power.py"),
            generator_runtime=Path("dualstream/generator.py"),
            schedule_implementation=Path("dualstream/compact_evidence.py"),
            freeze_validator=Path("scripts/h4c_freeze_manifest.py"),
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


def test_runtime_config_requires_explicit_device():
    runtime = _runtime()
    runtime["device"] = None
    with pytest.raises(ValueError, match="explicit nonempty string"):
        validate_runtime_config(runtime)


def test_freeze_manifest_rejects_underpowered_plan(tmp_path, monkeypatch):
    _manifest, _manifest_path = _make_frozen_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="underpowered"):
        build_manifest(
            git_commit="0" * 40,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            model_revision="immutable-revision",
            number_of_pairs=6,
            planning_p10=0.225,
            planning_p01=0.075,
            alpha_one_sided=0.05,
            target_power=0.90,
            preregistration=Path("docs/h4c_model_in_loop_preregistration.md"),
            scenario_set=Path("scenarios.json"),
            runtime_config=Path("runtime.json"),
            evaluator=Path("experiments/h4c_model_in_loop.py"),
            adapter=Path("experiments/h4c_local_hf_adapter.py"),
            runner=Path("experiments/h4c_local_model_confirmatory.py"),
            analysis=Path("experiments/h4c_paired_analysis.py"),
            power_helper=Path("experiments/h4c_power.py"),
            generator_runtime=Path("dualstream/generator.py"),
            schedule_implementation=Path("dualstream/compact_evidence.py"),
            freeze_validator=Path("scripts/h4c_freeze_manifest.py"),
            repo_root=tmp_path,
        )


def test_confirmatory_rejects_generation_limit_shorter_than_legal_outputs(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(_scenario_payload()), encoding="utf-8")
    scenarios = confirmatory.load_confirmatory_scenarios(path)
    runtime = _runtime()
    runtime["decoding"]["max_new_tokens"] = 5
    with pytest.raises(ValueError, match="cannot represent every legal output"):
        confirmatory._validate_generation_capacity(runtime, scenarios)


def test_confirmatory_requires_resolved_model_revision():
    class FakeModel:
        def parameters(self):
            return iter([type("P", (), {"device": "cpu"})()])

    adapter = type(
        "A",
        (),
        {
            "resolved_revision": None,
            "generator": type(
                "G",
                (),
                {"device": "cpu", "model": FakeModel()},
            )(),
        },
    )()
    with pytest.raises(ValueError, match="could not be resolved"):
        confirmatory._validate_loaded_model_identity_and_device(
            adapter,
            model_revision="immutable-revision",
            device="cpu",
        )


def test_confirmatory_verifies_loaded_device():
    class FakeModel:
        def parameters(self):
            return iter([type("P", (), {"device": "cpu"})()])

    adapter = type(
        "A",
        (),
        {
            "resolved_revision": "immutable-revision",
            "generator": type(
                "G",
                (),
                {"device": "cuda", "model": FakeModel()},
            )(),
        },
    )()
    with pytest.raises(ValueError, match="generator device"):
        confirmatory._validate_loaded_model_identity_and_device(
            adapter,
            model_revision="immutable-revision",
            device="cpu",
        )
