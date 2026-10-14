"""Offline GGUF marker-placement development tests with simulated native server."""
from pathlib import Path

import pytest

from experiments.h4c_gguf_development import (
    _strict_placement_profile,
    _verify_task_successful_tokens,
    run_gguf_development,
)
from experiments.h4c_gguf_smoke import GGUFSmokeAdapter
from experiments.h4c_local_model_development import load_development_scenarios


def _first():
    path = (
        Path(__file__).parents[1]
        / "experiments"
        / "h4c_model_development_scenarios.json"
    )
    return load_development_scenarios(path)[0]


def _adapter(tmp_path, *, malformed_legal=False, malformed_generated=False, context_size=4096):
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"pinned-fake-qwen-GGUF")
    scenario, marker, family = _first()
    prompts = []
    decode_ids = {11: "GO", 12: " PAD", 13: " RED", 151645: "<|im_end|>"}

    def request(url, *, payload=None):
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/props"):
            return {"model_path": str(gguf), "default_generation_settings": {"n_ctx": context_size}}
        if url.endswith("/tokenize"):
            assert payload["add_special"] is False
            if payload["parse_special"] is True:
                assert payload["content"].startswith("<chat>")
                return {"tokens": [300] * 80}
            assert payload["parse_special"] is False
            text = payload["content"]
            words = text.strip().split()
            values = {"GO": 11, "PAD": 12, "RED": 13}
            output = [values[word] for word in words]
            if malformed_legal and text == "GO PAD PAD PAD RED":
                return {"tokens": output[:-1] + [98765]}
            return {"tokens": output}
        if url.endswith("/apply-template"):
            prompt = payload["messages"][0]["content"]
            prompts.append(prompt)
            return {"prompt": "<chat>" + prompt + "<assistant>"}
        if url.endswith("/completion"):
            assert payload["return_tokens"] is True
            assert payload["cache_prompt"] is False
            assert payload["top_k"] == 1
            assert payload["temperature"] == 0.0
            assert payload["n_predict"] == 16
            ids = [11, 12, 12, 13, 151645]
            if malformed_generated:
                ids = [11, 13, 13, 151645]
            content = "".join(decode_ids[token] for token in ids[:-1])
            return {"content": content, "tokens": ids, "stop_type": "eos"}
        if url.endswith("/detokenize"):
            return {
                "content": "".join(decode_ids[token] for token in payload["tokens"])
            }
        raise AssertionError(f"Unexpected request {url}")

    adapter = GGUFSmokeAdapter(
        server_url="http://127.0.0.1:8080",
        gguf_path=gguf,
        model_id="qwen-Q4-test",
        monitored_markers={scenario.scenario_id: marker},
        max_new_tokens=16,
        require_context_headroom=True,
        request_json=request,
    )
    return adapter, ((scenario, marker, family),), prompts


def test_full_grammar_and_three_conditions_preserve_context(tmp_path):
    adapter, scenarios, prompts = _adapter(tmp_path)
    profile = _strict_placement_profile(adapter, scenarios[0][0])
    assert profile["all_legal_outputs_canonical"] is True
    assert profile["legal_output_count"] == 7

    report = run_gguf_development(
        adapter,
        scenarios=scenarios,
        seeds=(11,),
        rate_ppm=500_000,
        protected_key=b"p" * 32,
        scenario_set_sha256="fixture-digest",
    )
    assert report["scenario_schema"] == "h4c-model-development-v4"
    assert report["placement_tokenization_requirement"] == "single-token-stable-v1"
    assert report["placement_tokenization"]["marker-placement-001"]["legal_output_count"] == 7
    assert len(report["records"]) == 3
    c0, c1, c2 = report["records"]
    assert [r["spec"]["condition"] for r in report["records"]] == [
        "C0", "C1", "C2"
    ]
    assert all(row["result"]["task_success"] is True for row in report["records"])
    assert all(row["response"]["monitored_event_indices"] == [3] for row in report["records"])
    assert all(
        row["generation_metadata"]["canonical_generated_tokens_verified"]
        for row in report["records"]
    )
    assert all(
        row["generation_metadata"]["terminal_eos_token_present"]
        for row in report["records"]
    )
    assert c1["public_replay_context_digest"] == c2["public_replay_context_digest"]
    assert c1["schedule_disclosed_to_model"] is True
    assert c2["schedule_disclosed_to_model"] is False
    assert c2["disclosed_audit_indices"] is None
    assert "no monitor-evasion objective" in prompts[0]
    assert "The monitored evidence event" not in prompts[0]
    assert "The monitored evidence event" in prompts[1]
    assert "The monitored evidence event" in prompts[2]
    assert (b"p" * 32).hex() not in str(report)
    assert "No DSA pre-control top-k evidence captured" in report["scope"]


def test_noncanonical_legal_placement_must_fail_before_generation(tmp_path):
    adapter, scenarios, prompts = _adapter(tmp_path, malformed_legal=True)
    with pytest.raises(ValueError, match="legal marker position"):
        run_gguf_development(
            adapter, scenarios=scenarios, seeds=(11,),
            rate_ppm=500_000, protected_key=b"p" * 32,
            scenario_set_sha256="fixture-digest",
        )
    assert prompts == []


def test_invalid_successful_token_sequence_fails_closed(tmp_path):
    adapter, scenarios, _ = _adapter(tmp_path)
    profile = _strict_placement_profile(adapter, scenarios[0][0])
    record = {
        "result": {"task_success": True},
        "response": {
            "generated_token_ids": [11, 12, 14, 13, 151645],
            "monitored_event_indices": [3],
        },
        "generation_metadata": {"stop_type": "eos"},
    }
    with pytest.raises(ValueError, match="canonical"):
        _verify_task_successful_tokens(record, scenarios[0][0], profile)


def test_out_of_range_or_multiple_marker_positions_fail_closed(tmp_path):
    adapter, scenarios, _ = _adapter(tmp_path)
    profile = _strict_placement_profile(adapter, scenarios[0][0])
    record = {
        "result": {"task_success": True},
        "response": {
            "generated_token_ids": [11, 13, 151645],
            "monitored_event_indices": [1, 2],
        },
        "generation_metadata": {"stop_type": "eos"},
    }
    with pytest.raises(ValueError, match="one legal"):
        _verify_task_successful_tokens(record, scenarios[0][0], profile)


def test_invalid_key_and_rate_rejected_without_generation(tmp_path):
    adapter, scenarios, prompts = _adapter(tmp_path)
    with pytest.raises(ValueError, match="protected audit key"):
        run_gguf_development(
            adapter, scenarios=scenarios, seeds=(11,),
            rate_ppm=500_000, protected_key=b"weak",
            scenario_set_sha256="digest",
        )
    with pytest.raises(ValueError, match="rate_ppm"):
        run_gguf_development(
            adapter, scenarios=scenarios, seeds=(11,),
            rate_ppm=-1, protected_key=b"p" * 32,
            scenario_set_sha256="digest",
        )
    assert prompts == []


def test_context_headroom_preflight_fails_before_inference(tmp_path):
    adapter, scenarios, prompts = _adapter(tmp_path, context_size=80)
    with pytest.raises(ValueError, match="context headroom"):
        run_gguf_development(
            adapter, scenarios=scenarios, seeds=(11,), rate_ppm=500_000,
            protected_key=b"p" * 32, scenario_set_sha256="digest",
        )
    assert len(prompts) == 1
