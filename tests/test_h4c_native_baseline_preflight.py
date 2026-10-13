"""Fast fake-model tests for the non-confirmatory native HF C0 preflight."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.h4c_local_model_smoke import _specs, load_scenarios
from experiments.h4c_model_in_loop import build_model_request
from experiments.h4c_native_baseline_preflight import (
    build_c0_prompt,
    run_native_c0,
)


def _fixture():
    scenarios = Path(__file__).parents[1] / "experiments" / "h4c_model_smoke_scenarios.json"
    scenario, marker = load_scenarios(scenarios)[0]
    c0, c1, _ = _specs(
        scenario.scenario_id,
        model_id="fake-model",
        model_revision="fake-commit",
        seed=11,
        rate_ppm=500_000,
    )
    return scenario, marker, build_model_request(c0, scenario, audit_schedule=()), build_model_request(c1, scenario, audit_schedule=(1, 3))


def test_native_preflight_prompt_mirrors_c0_adapter_template_without_monitor_hint():
    scenario, _marker, request, _c1_request = _fixture()
    prompt = build_c0_prompt(request, scenario)
    assert prompt == (
        f"{scenario.task_prompt}\n\n"
        f"{request.treatment_instruction}\n"
        "Return only the final task output."
    )
    assert "The monitored evidence event is emitting" not in prompt


def test_native_preflight_rejects_nonbaseline_condition():
    scenario, _marker, _c0_request, c1_request = _fixture()
    with pytest.raises(ValueError, match="C0 only"):
        build_c0_prompt(c1_request, scenario)


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, prompt, *, return_tensors):
        assert return_tensors == "pt"
        assert prompt.startswith("<chat>")
        self.prompt = prompt
        return {
            "input_ids": torch.tensor([[20, 21]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, ids, *, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert tuple(ids) == (30, 31)
        return "RESULT=OK STYLE=GREEN"


class _Model:
    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        assert kwargs["max_new_tokens"] == 48
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["no_repeat_ngram_size"] == 0
        assert kwargs["eos_token_id"] == [0, 2]
        assert tuple(kwargs["input_ids"][0].tolist()) == (20, 21)
        return torch.tensor([[20, 21, 30, 31]], dtype=torch.long)


class _Adapter:
    model_id = "fake-model"
    model_revision = "fake-commit"
    resolved_revision = "fake-commit"
    prompt_format = "chat-template-v1"

    def __init__(self):
        self.generator = SimpleNamespace(
            tokenizer=_Tokenizer(),
            model=_Model(),
            device="cpu",
            _render_prompt=lambda prompt: "<chat>" + prompt + "<assistant>",
            _get_stop_token_ids=lambda: {2, 0},
        )

    def _marker_positions(self, token_ids, marker, answer):
        assert marker == "STYLE=RED"
        return ()


def test_native_preflight_uses_native_greedy_and_evaluates_c0():
    scenario, marker, request, _ = _fixture()
    report = run_native_c0(
        _Adapter(), request, scenario, marker, max_new_tokens=48
    )
    assert report["condition"] == "C0"
    assert report["task_success"] is True
    assert report["task_failure_reason"] is None
    assert report["parsed_choice"] == "GREEN"
    assert report["generated_token_ids"] == [30, 31]
    assert report["inference_backend"] == "transformers.generate.greedy"
    assert report["raw_prompt_sha256"] != report["rendered_prompt_sha256"]
