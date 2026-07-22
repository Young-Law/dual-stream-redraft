import json
from pathlib import Path

import pytest

from dualstream.experiment import REGISTRY, validate_protocol, run_exp01, verify_hashes


def test_protocol_validates():
    data = validate_protocol("experiments/protocols/dsa_v2_10_protocol.yaml")
    assert data["dsa_specification_version"].startswith("v2.10-blocked")


def test_unknown_experiment_not_registered():
    assert "not_real" not in REGISTRY


def test_exp01_run_emits_manifest_and_hashes(tmp_path):
    run_dir = run_exp01(Path("experiments/protocols/dsa_v2_10_protocol.yaml"), tmp_path, token_count=12, repetitions=4)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["experiment_id"] == "exp01_conformance"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["token_count"] == 12
    assert metrics["conditions"]["DSA-CI-Lite"]["evidence_completeness_rate"] == 1.0
    assert verify_hashes(run_dir)


def test_protocol_missing_required_field_fails(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"protocol_id": "bad"}))
    with pytest.raises(ValueError, match="protocol missing required fields"):
        validate_protocol(p)
