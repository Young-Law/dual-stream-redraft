"""Hermetic tests for the llama.cpp GGUF C0 diagnostic: no server or model needed."""
from pathlib import Path

import pytest

from experiments.h4c_gguf_c0_preflight import (
    _base_url,
    _check_server_model_path,
    run_gguf_c0,
)
from experiments.h4c_local_model_smoke import load_scenarios


def _scenario():
    path = (
        Path(__file__).parents[1] / "experiments" / "h4c_model_smoke_scenarios.json"
    )
    return load_scenarios(path)[0]


def _transport(content: str, tokens: list[int], model_path: Path):
    calls = []
    prefixes = {
        (100,): "RESULT=OK",
        (100, 101): "RESULT=OK STYLE=",
        (100, 101, 102): "RESULT=OK STYLE=RED",
    }

    def request(url, *, payload=None):
        calls.append((url, payload))
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/props"):
            return {"model_path": str(model_path)}
        if url.endswith("/apply-template"):
            assert payload["messages"][0]["role"] == "user"
            assert "no monitor-evasion objective" in payload["messages"][0]["content"]
            assert "The monitored evidence event" not in payload["messages"][0]["content"]
            return {"prompt": "<chat>" + payload["messages"][0]["content"] + "<assistant>"}
        if url.endswith("/completion"):
            assert payload["return_tokens"] is True
            assert payload["cache_prompt"] is False
            assert payload["temperature"] == 0.0
            assert payload["top_k"] == 1
            assert payload["seed"] == 11
            assert payload["n_predict"] == 48
            return {"content": content, "tokens": tokens, "stop_type": "eos"}
        if url.endswith("/detokenize"):
            return {"content": prefixes[tuple(payload["tokens"])]}
        raise AssertionError(url)

    return request, calls


def test_gguf_c0_preserves_generated_token_positions_and_model_sha(tmp_path):
    model_path = tmp_path / "fake.gguf"
    model_path.write_bytes(b"quantized checkpoint bytes")
    scenario, marker = _scenario()
    request, calls = _transport(
        "RESULT=OK STYLE=RED", [100, 101, 102], model_path
    )
    report = run_gguf_c0(
        server_url="http://127.0.0.1:8080",
        gguf_path=model_path,
        model_id="test-qwen-gguf",
        scenario=scenario,
        monitored_marker=marker,
        seed=11,
        rate_ppm=500_000,
        max_new_tokens=48,
        request_json=request,
    )
    assert report["task_success"] is True
    assert report["parsed_choice"] == "RED"
    assert report["generated_token_ids"] == [100, 101, 102]
    assert report["monitored_event_indices"] == [1, 2]
    assert report["gguf_sha256"] == report["model_revision"]
    assert len(report["gguf_sha256"]) == 64
    assert [url.rsplit("/", 1)[-1] for url, _ in calls].count("detokenize") == 3


def test_gguf_c0_rejects_unavailable_token_ids(tmp_path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"fake")
    scenario, marker = _scenario()
    request, _ = _transport("RESULT=OK STYLE=RED", [], model)
    with pytest.raises(ValueError, match="detokenization differs"):
        run_gguf_c0(
            server_url="http://localhost:8080",
            gguf_path=model, model_id="fake-model", scenario=scenario,
            monitored_marker=marker, seed=11, rate_ppm=500_000,
            max_new_tokens=48, request_json=request,
        )


def test_gguf_model_identity_mismatch_fails_closed(tmp_path):
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    with pytest.raises(ValueError, match="different GGUF"):
        _check_server_model_path({"model_path": str(first)}, second)


@pytest.mark.parametrize("bad", [
    "https://127.0.0.1:8080",
    "http://192.168.1.20:8080",
    "http://example.com:8080",
    "http://127.0.0.1:8080/custom",
    "http://user:password@localhost:8080",
])
def test_gguf_preflight_restricts_requests_to_local_http(bad):
    with pytest.raises(ValueError, match="loopback"):
        _base_url(bad)
