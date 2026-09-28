from types import SimpleNamespace

import pytest
import torch

from dualstream.generator import DualStreamGenerator, GenerationConfig


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 0
    additional_special_tokens_ids = []

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.ones(1, 1)}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join("<|endoftext|>" if i == 0 else f"word{i}" for i in ids)


def generator(logits):
    gen = DualStreamGenerator.__new__(DualStreamGenerator)
    gen.model_name = "deterministic-fixture"
    gen.device = "cpu"
    gen.tokenizer = Tokenizer()
    gen.model = lambda **kwargs: SimpleNamespace(
        logits=torch.tensor([[logits]], dtype=torch.float32),
        past_key_values=None, attentions=None, hidden_states=None,
    )
    return gen


def config(**kwargs):
    return GenerationConfig(max_new_tokens=4, top_k=3, do_sample=False,
                            enable_heuristics=False, audit_mode="off", **kwargs)


def test_eos_stops_when_also_used_as_padding_and_bos():
    gen = generator([10.0, 0.0, 0.0])
    assert gen._get_stop_token_ids() == {0}
    result = gen.generate("hello", config())
    assert len(result["frames"]) == 1
    assert result["frames"][0].chosen_id == 0


@pytest.mark.parametrize("top_p, expected", [
    (0.8, [2/3, 1/3, 0]), (0.6, [1, 0, 0]),
    (0.1, [1, 0, 0]), (1.0, [0.6, 0.3, 0.1]),
])
def test_top_p_retains_threshold_crossing_token(top_p, expected):
    actual = DualStreamGenerator._apply_top_p(torch.tensor([0.6, 0.3, 0.1]), top_p)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32))


def test_repetition_penalty_changes_choice_without_changing_evidence():
    gen = generator([-10.0, 4.0, 3.0])
    plain = gen.generate("hello", config())
    controlled = gen.generate("hello", config(repetition_penalty=2.0))
    assert plain["frames"][0].chosen_id == 1
    assert controlled["frames"][0].chosen_id == 2
    assert controlled["frames"][0].topk == plain["frames"][0].topk
    assert "repetition_penalty" in controlled["frames"][0].decode_controls_applied


def test_ngram_control_uses_full_history_across_cached_steps():
    gen = generator([-10.0, 4.0, 3.0])
    result = gen.generate("hello", config(no_repeat_ngram_size=2))
    ids = [1] + [frame.chosen_id for frame in result["frames"]]
    bigrams = list(zip(ids, ids[1:]))
    assert len(bigrams) == len(set(bigrams))
    assert all("no_repeat_ngram" in frame.decode_controls_applied for frame in result["frames"])


@pytest.mark.parametrize("options", [
    {"max_new_tokens": 0}, {"top_k": 0}, {"temperature": 0},
    {"top_p": 0}, {"top_p": 1.1}, {"repetition_penalty": float("nan")},
    {"repetition_penalty": -1}, {"no_repeat_ngram_size": -1},
])
def test_invalid_generation_settings_are_rejected(options):
    with pytest.raises(ValueError):
        GenerationConfig(**options)
