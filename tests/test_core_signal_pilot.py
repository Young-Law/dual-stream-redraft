"""No model downloads: regression tests for source-controlled pilot scoring."""
import json

import pytest

from experiments.core_signal_pilot import (
    _auc,
    _pair_delta,
    fixture,
    prepare,
)


def test_fixture_is_deterministic_and_balanced():
    a = fixture()
    assert a == fixture()
    assert a["task_count"] == 48
    assert a["problem_count"] == 12
    assert sum(row["claim_truth"] for row in a["tasks"]) == 24
    assert sum(row["framing"] == "pressure" for row in a["tasks"]) == 24
    assert len({row["prompt"] for row in a["tasks"]}) == 48
    for problem_id in range(1, 13):
        group = [row for row in a["tasks"] if row["problem_id"] == problem_id]
        assert len(group) == 4
        assert {(row["claim_truth"], row["framing"]) for row in group} == {
            (truth, framing)
            for truth in (True, False)
            for framing in ("neutral", "pressure")
        }


def test_prepare_produces_same_order_and_binds_hash(tmp_path):
    prepare(tmp_path)
    prompts = (tmp_path / "prompts.txt").read_text(encoding="utf-8").splitlines()
    spec = json.loads((tmp_path / "fixture.json").read_text(encoding="utf-8"))
    assert prompts == [row["prompt"] for row in spec["tasks"]]
    assert len((tmp_path / "fixture.sha256").read_text().strip()) == 64


def test_auc_is_directional_and_handles_one_class():
    assert _auc([(0.9, True), (0.2, False)]) == 1.0
    assert _auc([(0.1, True), (0.8, False)]) == 0.0
    assert _auc([(0.5, True), (0.5, False)]) == 0.5
    assert _auc([(0.5, True), (0.2, True)]) is None


def test_paired_false_claim_outcomes_have_correct_denominator():
    tasks = fixture()["tasks"]
    rows = [
        {
            **task,
            "parsed_answer": (
                "YES" if not task["claim_truth"] and task["framing"] == "pressure"
                else "NO"
            ),
        }
        for task in tasks
    ]
    pressured, neutral, effect = _pair_delta(rows)
    assert (pressured, neutral, effect) == (12, 0, 1.0)
    with pytest.raises(ValueError, match="incomplete paired"):
        _pair_delta([r for r in rows if not (r["problem_id"] == 1 and
                                            r["framing"] == "pressure" and
                                            not r["claim_truth"])])
