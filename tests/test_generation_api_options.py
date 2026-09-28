from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

from dualstream.api import app
from dualstream.service import DualStreamService


@pytest.fixture
def generation_client(monkeypatch, tmp_path):
    service = DualStreamService()
    monkeypatch.setattr("dualstream.api.service", service)
    monkeypatch.setattr("dualstream.service.DualStreamGenerator", lambda *a, **kw: object())
    observed = {}

    def capture(gen, cfg, prompt, outdir):
        observed.update(asdict(cfg))

    monkeypatch.setattr("dualstream.service._run_generation", capture)
    monkeypatch.setattr("dualstream.service.load_generation_artifacts", lambda _: observed.copy())
    with TestClient(app) as client:
        def generate(options):
            response = client.post("/generate", json={
                "prompt": "hello", "offline": False, "outdir": str(tmp_path), **options,
            })
            assert response.status_code == 200
            service._executor.shutdown(wait=True)
            return client.get("/jobs/" + response.json()["job_id"]).json()
        yield generate
    service._executor.shutdown(wait=True)


@pytest.mark.parametrize("wire_version", ["0x0302", "0x0303", 0x0303])
def test_generate_forwards_evidence_and_audit_options(generation_client, wire_version):
    options = {
        "compact_evidence": True, "adaptive_k": True,
        "evidence_profile": "DSA-CI-Standard", "top_k": 1,
        "max_adaptive_k": 9, "chunk_token_capacity": 64,
        "compact_wire_version": wire_version, "audit_mode": "full",
        "poc_mode": "level1_sycophancy_proxy", "randomized_audit": True,
        "audit_nonce": 123, "entropy_threshold": 3.0, "refusal_mass_threshold": 0.1,
        "risk_threshold_review": 0.3, "risk_threshold_fail": 0.6,
        "max_red_retries": 2, "fallback_strategy": "abort", "no_selective_retention": True,
        "repetition_penalty": 1.15, "no_repeat_ngram_size": 3,
    }
    job = generation_client(options)
    assert job["status"] == "completed", job["error"]
    cfg = job["result"]
    for key, value in options.items():
        if key not in {"top_k", "compact_wire_version", "no_selective_retention"}:
            assert cfg[key] == value, key
    assert cfg["top_k"] == 5  # CI-Standard's minimum, as on the CLI
    assert cfg["compact_wire_version"] == (int(wire_version, 0) if isinstance(wire_version, str) else wire_version)
    assert cfg["selective_retention"] is False


def test_generate_preserves_legacy_defaults(generation_client):
    job = generation_client({})
    assert job["status"] == "completed", job["error"]
    assert job["result"]["compact_evidence"] is False
    assert job["result"]["adaptive_k"] is False
    assert job["result"]["max_adaptive_k"] is None
    assert job["result"]["compact_wire_version"] == 0x0303


def test_generate_rejects_unsupported_wire_version(generation_client):
    job = generation_client({"compact_evidence": True, "compact_wire_version": "0x9999"})
    assert job["status"] == "failed"
    assert "compact_wire_version must be" in job["error"]
