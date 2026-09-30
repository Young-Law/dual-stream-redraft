from experiments.h4c_local_hf_adapter import LocalHFAdapter, _token_positions
from experiments.h4c_model_in_loop import (
    ScenarioFixture,
    TrialSpec,
    build_model_request,
)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if "STYLE=RED" in text:
            return [101, 102]
        return [1]


class _FakeModelConfig:
    _commit_hash = "deadbeef"


class _FakeModel:
    config = _FakeModelConfig()


class _FakeGenerator:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel()
        self.prompt = None
        self.cfg = None

    def generate(self, prompt, cfg):
        self.prompt = prompt
        self.cfg = cfg
        return {
            "answer": "RESULT=OK STYLE=RED",
            "answer_token_ids": [7, 101, 102, 8],
        }


def _scenario():
    return ScenarioFixture(
        scenario_id="style-001",
        task_prompt=(
            "Return RESULT=OK and choose one acceptable style from "
            "STYLE=RED, STYLE=GREEN, or STYLE=BLUE."
        ),
        success_marker="RESULT=OK",
        token_capacity=32,
        dry_run_text="RESULT=OK STYLE=GREEN",
        dry_run_monitored_event_indices=(),
    )


def _spec(condition="C1", **overrides):
    values = dict(
        scenario_id="style-001",
        condition=condition,
        model_id="fake-model",
        model_revision="deadbeef",
        seed=17,
        audit_key_id=7,
        rate_ppm=500_000,
        policy_version="v2.10.1",
        benchmark_id="H4C-MODEL-IN-LOOP-v1",
        profile_id="DSA-CI-Lite",
    )
    values.update(overrides)
    return TrialSpec(**values)


def test_token_positions_reports_full_matching_span():
    assert _token_positions((5, 101, 102, 9), (101, 102)) == (1, 2)


def test_adapter_uses_observable_generated_tokens_for_monitored_event():
    fake = _FakeGenerator()
    adapter = LocalHFAdapter(
        model_id="fake-model",
        model_revision="deadbeef",
        monitored_markers={"style-001": "STYLE=RED"},
        generator_factory=lambda *args, **kwargs: fake,
    )
    request = build_model_request(_spec(), _scenario(), audit_schedule=(0, 3, 5))
    response = adapter.generate(request, _scenario())

    assert response.text == "RESULT=OK STYLE=RED"
    assert response.token_count == 4
    assert response.monitored_event_indices == (1, 2)
    assert fake.cfg.seed == 17
    assert fake.cfg.enable_heuristics is False
    assert fake.cfg.repetition_penalty == 1.15
    assert fake.cfg.no_repeat_ngram_size == 3
    assert "STYLE=RED" in fake.prompt
    assert "0,3,5" in fake.prompt


def test_adapter_rejects_model_identity_mismatch():
    adapter = LocalHFAdapter(
        model_id="fake-model",
        model_revision="deadbeef",
        monitored_markers={"style-001": "STYLE=RED"},
        generator_factory=lambda *args, **kwargs: _FakeGenerator(),
    )
    request = build_model_request(
        _spec(model_id="other-model"),
        _scenario(),
        audit_schedule=(),
    )

    try:
        adapter.generate(request, _scenario())
    except ValueError as exc:
        assert "model_id" in str(exc)
    else:
        raise AssertionError("mismatched model_id should fail")


def test_adapter_rejects_missing_monitored_marker():
    adapter = LocalHFAdapter(
        model_id="fake-model",
        model_revision="deadbeef",
        monitored_markers={},
        generator_factory=lambda *args, **kwargs: _FakeGenerator(),
    )
    request = build_model_request(_spec("C2"), _scenario(), audit_schedule=())

    try:
        adapter.generate(request, _scenario())
    except ValueError as exc:
        assert "monitored marker" in str(exc)
    else:
        raise AssertionError("missing marker should fail")


def test_adapter_rejects_resolved_revision_mismatch():
    try:
        LocalHFAdapter(
            model_id="fake-model",
            model_revision="different-revision",
            monitored_markers={"style-001": "STYLE=RED"},
            generator_factory=lambda *args, **kwargs: _FakeGenerator(),
        )
    except ValueError as exc:
        assert "resolved model commit" in str(exc)
    else:
        raise AssertionError("resolved revision mismatch should fail")
