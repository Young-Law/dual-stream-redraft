"""No model download: verify isolation and recording in prompt calibration."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.h4c_local_model_smoke import _specs, load_scenarios
from experiments.h4c_prompt_calibration import (
    VARIANTS,
    calibration_prompts,
    run_calibration,
)


def fixture():
    path = Path(__file__).parents[1] / "experiments" / "h4c_model_smoke_scenarios.json"
    scenario, marker = load_scenarios(path)[0]
    c0, c1, _ = _specs(
        scenario.scenario_id,
        model_id="fake-model",
        model_revision="fake-commit",
        seed=11,
        rate_ppm=500_000,
    )
    return scenario, marker, c0, c1


def test_prompt_calibration_preserves_original_scenario_and_is_labeled():
    scenario, _, _, _ = fixture()
    prompts = calibration_prompts(
        scenario.task_prompt, scenario.success_marker,
        scenario.choice_field, scenario.allowed_choice_values,
    )
    assert set(prompts) == set(VARIANTS)
    assert prompts["task-only"].startswith(scenario.task_prompt)
    assert "monitor-evasion" not in prompts["task-only"]
    assert "RESULT=OK STYLE=RED" in prompts["copy-target"]
    assert "exactly" in prompts["copy-target"]


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        assert prompt.startswith("<chat>")
        return {
            "input_ids": torch.tensor([[20, 21]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        assert tuple(ids) == (30, 31)
        return "RESULT=OK STYLE=GREEN"


class FakeModel:
    def __init__(self):
        self.calls = 0

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 24
        assert kwargs["no_repeat_ngram_size"] == 0
        assert kwargs["repetition_penalty"] == 1.0
        assert kwargs["eos_token_id"] == [0, 2]
        self.calls += 1
        return torch.tensor([[20, 21, 30, 31]])


class FakeAdapter:
    model_id = "fake-model"
    model_revision = "fake-commit"
    resolved_revision = "fake-commit"

    def __init__(self):
        self.model = FakeModel()
        self.generator = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=self.model,
            device="cpu",
            _render_prompt=lambda prompt: "<chat>" + prompt + "<assistant>",
            _get_stop_token_ids=lambda: {0, 2},
        )

    def _marker_positions(self, ids, marker, text):
        assert marker == "STYLE=RED"
        return ()


def test_two_variants_run_with_one_adapter_and_evaluate_only_c0():
    scenario, marker, c0, _ = fixture()
    adapter = FakeAdapter()
    report = run_calibration(
        adapter, scenario, marker, c0, variants=VARIANTS, max_new_tokens=24
    )
    assert adapter.model.calls == 2
    assert report["scope"].startswith("Non-confirmatory C0 prompt calibration")
    assert len(report["records"]) == 2
    assert [r["variant"] for r in report["records"]] == list(VARIANTS)
    assert all(r["task_success"] for r in report["records"])
    assert all(r["parsed_choice"] == "GREEN" for r in report["records"])
    assert report["records"][0]["prompt_sha256"] != report["records"][1]["prompt_sha256"]


def test_calibration_is_restricted_to_c0_and_known_variants():
    scenario, marker, c0, c1 = fixture()
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="C0"):
        run_calibration(adapter, scenario, marker, c1, variants=VARIANTS, max_new_tokens=24)
    with pytest.raises(ValueError, match="unknown"):
        run_calibration(adapter, scenario, marker, c0, variants=("not-a-variant",), max_new_tokens=24)
    with pytest.raises(ValueError, match="positive"):
        run_calibration(adapter, scenario, marker, c0, variants=VARIANTS, max_new_tokens=0)
