import pytest

from experiments.h4c_model_in_loop import TrialResult, TrialSpec, assert_paired_public_context


def _spec(condition: str, **overrides) -> TrialSpec:
    values = dict(
        scenario_id="scenario-001",
        condition=condition,
        model_id="dry-run-model",
        model_revision="frozen",
        seed=17,
        audit_key_id=7,
        rate_ppm=5000,
        policy_version="v2.10",
        benchmark_id="H4C-MODEL-IN-LOOP-v1",
        profile_id="default",
    )
    values.update(overrides)
    return TrialSpec(**values)


def test_h4c_pair_accepts_identical_public_context():
    assert_paired_public_context(_spec("C1"), _spec("C2"))


def test_h4c_pair_rejects_audit_key_id_confound():
    with pytest.raises(ValueError, match="audit_key_id"):
        assert_paired_public_context(_spec("C1"), _spec("C2", audit_key_id=8))


def test_successful_adaptive_evasion_requires_success_without_exposure():
    result = TrialResult(_spec("C2"), task_success=True, audit_exposed=False, refused=False, invalid=False)
    assert result.successful_adaptive_evasion is True
