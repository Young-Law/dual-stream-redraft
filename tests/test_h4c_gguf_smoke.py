"""Mocked-server GGUF H4-C paired smoke tests; no downloaded model required."""
from pathlib import Path

import pytest

from experiments.h4c_gguf_smoke import GGUFSmokeAdapter, run_gguf_smoke
from experiments.h4c_local_model_smoke import load_scenarios


def _scenario():
    fixture = (
        Path(__file__).parents[1] / "experiments" / "h4c_model_smoke_scenarios.json"
    )
    return load_scenarios(fixture)[0]


def test_gguf_paired_smoke_keeps_C1_C2_context_and_protected_key_private(tmp_path):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"fake-quantized-model")
    scenario, marker = _scenario()
    prompts = []
    token_prefixes = {
        (100,): "RESULT=OK ",
        (100, 101): "RESULT=OK STYLE=",
        (100, 101, 102): "RESULT=OK STYLE=RED",
    }

    def fake_request(url, *, payload=None):
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/props"):
            return {"model_path": str(gguf)}
        if url.endswith("/apply-template"):
            raw = payload["messages"][0]["content"]
            prompts.append(raw)
            return {"prompt": "<chat>" + raw + "<assistant>"}
        if url.endswith("/completion"):
            assert payload["return_tokens"] is True
            assert payload["cache_prompt"] is False
            assert payload["top_k"] == 1
            assert payload["temperature"] == 0.0
            if "The monitored evidence event" in payload["prompt"]:
                return {
                    "content": "RESULT=OK STYLE=RED",
                    "tokens": [100, 101, 102],
                    "stop_type": "eos",
                }
            return {
                "content": "RESULT=OK STYLE=GREEN",
                "tokens": [200, 201, 202],
                "stop_type": "eos",
            }
        if url.endswith("/detokenize"):
            return {"content": token_prefixes[tuple(payload["tokens"])]}
        raise AssertionError(url)

    adapter = GGUFSmokeAdapter(
        server_url="http://127.0.0.1:8080",
        gguf_path=gguf,
        model_id="qwen-test-Q4",
        monitored_markers={scenario.scenario_id: marker},
        request_json=fake_request,
    )
    report = run_gguf_smoke(
        adapter,
        scenarios=((scenario, marker),),
        seed=11,
        rate_ppm=500_000,
        protected_key=b"p" * 32,
    )
    assert len(report["records"]) == 3
    c0, c1, c2 = report["records"]
    assert [row["spec"]["condition"] for row in report["records"]] == [
        "C0", "C1", "C2"
    ]
    assert c0["result"]["task_success"] is True
    assert c1["result"]["task_success"] is True
    assert c2["result"]["task_success"] is True
    assert c1["response"]["generated_token_ids"] == [100, 101, 102]
    assert c1["response"]["monitored_event_indices"] == [1, 2]
    assert c0["response"]["monitored_event_indices"] == []
    assert c1["public_replay_context_digest"] == c2["public_replay_context_digest"]
    assert c1["schedule_disclosed_to_model"] is True
    assert c2["schedule_disclosed_to_model"] is False
    assert c1["disclosed_audit_indices"] is not None
    assert c2["disclosed_audit_indices"] is None
    assert "The monitored evidence event" not in prompts[0]
    assert all("The monitored evidence event" in prompt for prompt in prompts[1:])
    assert "H4C_PROTECTED_KEY_HEX" not in str(report)
    assert (b"p" * 32).hex() not in str(report)
    assert report["gguf_sha256"] == adapter.model_revision
    assert "No DSA pre-control top-k evidence captured" in report["scope"]


def test_gguf_paired_smoke_fails_closed_for_invalid_keys_and_rates(tmp_path):
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"model")
    scenario, marker = _scenario()

    def fake_request(url, *, payload=None):
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/props"):
            return {"model_path": str(gguf)}
        raise AssertionError("unexpected inference request")

    adapter = GGUFSmokeAdapter(
        server_url="http://127.0.0.1:8080",
        gguf_path=gguf,
        model_id="qwen-test-Q4",
        monitored_markers={scenario.scenario_id: marker},
        request_json=fake_request,
    )
    with pytest.raises(ValueError, match="invalid protected audit key"):
        run_gguf_smoke(
            adapter, scenarios=((scenario, marker),),
            seed=11, rate_ppm=500_000, protected_key=b"tiny",
        )
    with pytest.raises(ValueError, match="rate_ppm"):
        run_gguf_smoke(
            adapter, scenarios=((scenario, marker),),
            seed=11, rate_ppm=1_000_001, protected_key=b"p" * 32,
        )
