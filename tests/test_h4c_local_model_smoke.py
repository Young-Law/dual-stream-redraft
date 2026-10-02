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
