"""No-model GGUF tokenizer development preflight tests."""
from pathlib import Path

import pytest

from experiments.h4c_gguf_tokenizer_preflight import (
    _tokenize,
    inspect_scenario_tokenization,
    run_gguf_tokenizer_preflight,
)
from experiments.h4c_local_model_development import load_development_scenarios


def _first_scenario():
    path = Path(__file__).parents[1] / "experiments" / "h4c_model_development_scenarios.json"
    return load_development_scenarios(path)[0]


def _transport(path: Path, *, incompatible=False):
    calls = []
    vocab = {
        "GO": [11],
        " PAD": [12],
        " RED": [13],
        "GO PAD RED": [11, 12, 13],
        "GO PAD PAD RED": (
            [11, 12, 25, 13] if incompatible else [11, 12, 12, 13]
        ),
    }

    def request(url, *, payload=None):
        calls.append((url, payload))
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/props"):
            return {"model_path": str(path)}
        if url.endswith("/tokenize"):
            assert payload["add_special"] is False
            assert payload["parse_special"] is False
            return {"tokens": vocab[payload["content"]]}
        raise AssertionError(f"Unexpected endpoint: {url}")

    return request, calls


def test_one_token_stable_development_grammar(tmp_path):
    path = tmp_path / "qwen.gguf"
    path.write_bytes(b"pinned-Q4-test")
    scenario, marker, family = _first_scenario()
    request, calls = _transport(path)
    report = run_gguf_tokenizer_preflight(
        server_url="http://127.0.0.1:8080",
        gguf_path=path,
        model_id="Qwen2.5-3B-Instruct-Q4_K_M",
        scenarios=((scenario, marker, family),),
        scenario_set_sha256="fixture-digest",
        request_json=request,
    )
    assert report["all_scenarios_tokenization_compatible"] is True
    assert report["checked_scenario_count"] == 1
    assert report["incompatible_scenarios"] == []
    assert report["placement_tokenization"][scenario.scenario_id]["canonical_sequence_stable"] is True
    assert report["placement_tokenization"][scenario.scenario_id]["monitored_token_ids"] == [13]
    assert report["gguf_sha256"] == report["model_revision"]
    assert len([call for call, _ in calls if call.endswith("/tokenize")]) == 5
    assert "Non-confirmatory" in report["scope"]


def test_context_dependent_tokenization_fails_preflight(tmp_path):
    path = tmp_path / "qwen.gguf"
    path.write_bytes(b"pinned-Q4-test")
    scenario, marker, family = _first_scenario()
    request, _ = _transport(path, incompatible=True)
    report = run_gguf_tokenizer_preflight(
        server_url="http://localhost:8080", gguf_path=path,
        model_id="qwen", scenarios=((scenario, marker, family),),
        scenario_set_sha256="digest", request_json=request,
    )
    assert report["all_scenarios_tokenization_compatible"] is False
    assert report["incompatible_scenarios"] == [scenario.scenario_id]
    assert report["placement_tokenization"][scenario.scenario_id]["canonical_sequence_stable"] is False


def test_missing_token_ids_rejected():
    with pytest.raises(ValueError, match="malformed token IDs"):
        _tokenize(
            "http://127.0.0.1:8080", "GO",
            lambda *_args, **_kwargs: {"tokens": [True]},
        )


def test_wrong_model_file_rejected(tmp_path):
    path = tmp_path / "qwen.gguf"
    path.write_bytes(b"correct")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"incorrect")
    scenario, marker, family = _first_scenario()
    request, calls = _transport(other)
    with pytest.raises(ValueError, match="different GGUF"):
        run_gguf_tokenizer_preflight(
            server_url="http://127.0.0.1:8080",
            gguf_path=path, model_id="qwen",
            scenarios=((scenario, marker, family),),
            scenario_set_sha256="digest", request_json=request,
        )
