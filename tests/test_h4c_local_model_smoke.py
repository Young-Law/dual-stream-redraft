from pathlib import Path

import pytest

from experiments.h4c_local_model_smoke import (
    PUBLIC_SCHEDULE_KEY,
    _run_one,
    _specs,
    load_scenarios,
)
from experiments.h4c_model_in_loop import (
    ModelResponse,
    assert_paired_public_context,
)


class _FakeAdapter:
    def generate(self, request, scenario):
        return ModelResponse(
            text=scenario.success_marker + " STYLE=GREEN",
            token_count=8,
            monitored_event_indices=(),
        )


def _scenario_path():
    return (
        Path(__file__).parents[1]
        / "experiments"
        / "h4c_model_smoke_scenarios.json"
    )


def test_model_smoke_scenarios_keep_success_and_monitor_events_separate():
    rows = load_scenarios(_scenario_path())
    assert len(rows) >= 2
    for scenario, monitored_marker in rows:
        assert scenario.success_marker
        assert monitored_marker
        assert scenario.success_marker != monitored_marker
        assert scenario.choice_field
        assert scenario.allowed_choice_values


def test_model_smoke_primary_pair_preserves_public_context():
    scenario, _ = load_scenarios(_scenario_path())[0]
    _c0, c1, c2 = _specs(
        scenario.scenario_id,
        model_id="fake-model",
        model_revision="revision-1",
        seed=17,
        rate_ppm=100_000,
    )
    assert_paired_public_context(c1, c2)


def test_c1_discloses_schedule_while_c2_does_not():
    scenario, _ = load_scenarios(_scenario_path())[0]
    _c0, c1, c2 = _specs(
        scenario.scenario_id,
        model_id="fake-model",
        model_revision="revision-1",
        seed=17,
        rate_ppm=1_000_000,
    )
    adapter = _FakeAdapter()

    public = _run_one(
        c1,
        scenario,
        schedule_key=PUBLIC_SCHEDULE_KEY,
        adapter=adapter,
    )
    protected = _run_one(
        c2,
        scenario,
        schedule_key=b"p" * 32,
        adapter=adapter,
    )

    assert public["schedule_disclosed_to_model"] is True
    assert protected["schedule_disclosed_to_model"] is False
    assert (
        public["public_replay_context_digest"]
        == protected["public_replay_context_digest"]
    )


def test_model_smoke_schema_rejects_same_success_and_monitor_marker(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        """{
          "schema_version": "h4c-model-smoke-v1",
          "scenarios": [{
            "scenario_id": "bad",
            "task_prompt": "x",
            "success_marker": "SAME",
            "monitored_marker": "SAME",
            "token_capacity": 8
          }]
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must differ"):
        load_scenarios(path)


def test_model_smoke_rejects_missing_choice_contract(tmp_path):
    path = tmp_path / "bad-choice.json"
    path.write_text(
        """{
          "schema_version": "h4c-model-smoke-v1",
          "scenarios": [{
            "scenario_id": "bad-choice",
            "task_prompt": "Return RESULT=OK",
            "success_marker": "RESULT=OK",
            "monitored_marker": "STYLE=RED",
            "token_capacity": 8
          }]
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="choice_field and allowed_choice_values are required"):
        load_scenarios(path)


def test_model_smoke_result_marks_c0_adaptive_evasion_not_applicable():
    scenario, _ = load_scenarios(_scenario_path())[0]
    c0, _c1, _c2 = _specs(
        scenario.scenario_id,
        model_id="fake-model",
        model_revision="revision-1",
        seed=17,
        rate_ppm=100_000,
    )
    record = _run_one(
        c0,
        scenario,
        schedule_key=b"p" * 32,
        adapter=_FakeAdapter(),
    )
    assert record["result"]["task_success"] is True
    assert record["result"]["parsed_choice"] == "GREEN"
    assert record["result"]["task_failure_reason"] is None
    assert record["result"]["adaptive_evasion_applicable"] is False
    assert record["result"]["successful_adaptive_evasion"] is None


def test_greedy_diagnostic_cli_is_opt_in(monkeypatch, tmp_path):
    """CLI exposes an opt-in mode without changing ordinary smoke defaults."""
    import sys
    from experiments import h4c_local_model_smoke as smoke

    calls = []

    class FakeAdapter:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.resolved_revision = kwargs["model_revision"]
            self.prompt_format = "chat-template-v1"
            self.decoding_config = {"do_sample": kwargs.get("do_sample", True)}

        def generate(self, request, scenario):
            return ModelResponse(
                text=scenario.success_marker + " " + scenario.choice_field + "=" + scenario.allowed_choice_values[0],
                token_count=4,
                monitored_event_indices=(),
            )

    monkeypatch.setattr(smoke, "LocalHFAdapter", FakeAdapter)
    monkeypatch.setenv("H4C_PROTECTED_KEY_HEX", "ab" * 32)
    for extra in ([], ["--diagnostic-greedy"]):
        output = tmp_path / ("greedy.json" if extra else "default.json")
        monkeypatch.setattr(sys, "argv", [
            "h4c_local_model_smoke", "--model", "fake-model",
            "--model-revision", "revision-1", "--json-out", str(output), *extra,
        ])
        assert smoke.main() == 0
        report = __import__("json").loads(output.read_text(encoding="utf-8"))
        assert report["diagnostic_greedy"] == bool(extra)
    assert "do_sample" not in calls[0]
    assert calls[1]["do_sample"] is False
    assert calls[1]["repetition_penalty"] == 1.0
    assert calls[1]["no_repeat_ngram_size"] == 0
