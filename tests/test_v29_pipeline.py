from dualstream.compact_evidence import encode_compact_sequence, reconstruct_token_evidence
from dualstream.verifier import verify_evidence_artifact


def test_v29_pipeline_compact_passes(tmp_path):
    rows = [{"chosen_id": i+10, "topk_ids": [i+10, i+20, i+30], "topk_scores": [.6,.3,.1]} for i in range(1000)]
    blob = encode_compact_sequence(rows, profile="DSA-CI-Lite", adaptive_k=True, max_adaptive_k=5)
    assert len(reconstruct_token_evidence(blob)) == len(rows)
    (tmp_path / "compact_evidence.dsae").write_bytes(blob)
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert report.ok, report.errors


def test_generator_fallback_is_separate_from_evidence_bound_answer(monkeypatch):
    import types
    import torch
    from dualstream.generator import DualStreamGenerator, GenerationConfig
    from dualstream.compact_evidence import reconstruct_token_evidence

    class Tok:
        eos_token_id = None; pad_token_id = None; bos_token_id = None; additional_special_tokens_ids = None
        def __call__(self, _prompt, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        def decode(self, tokens, skip_special_tokens=False):
            return "tok" if tokens else ""
    class Model:
        def __call__(self, **_kwargs):
            return types.SimpleNamespace(logits=torch.tensor([[[0.0, 5.0, 4.0, 3.0, 2.0]]]), past_key_values=None, attentions=None, hidden_states=None)
    monkeypatch.setattr("dualstream.generator.coherence_outcome", lambda *_args, **_kwargs: types.SimpleNamespace(outcome="FAIL"))
    gen = DualStreamGenerator.__new__(DualStreamGenerator)
    gen.model_name = "fake"; gen.device = "cpu"; gen.tokenizer = Tok(); gen.model = Model()
    cfg = GenerationConfig(max_new_tokens=1, top_k=3, do_sample=False, compact_evidence=True, enable_heuristics=False, include_running_hash=False)
    result = gen.generate("x", cfg)
    assert result["answer"] == "tok"
    assert result["fallback_text"]
    recs = reconstruct_token_evidence(result["compact_evidence_bytes"])
    assert recs[0]["chosen_id"] == result["answer_token_ids"][0]


def test_ci_standard_compact_generation_raises_capture_to_profile_base_k():
    import types
    import torch
    from dualstream.generator import DualStreamGenerator, GenerationConfig

    class Tok:
        eos_token_id = None; pad_token_id = None; bos_token_id = None; additional_special_tokens_ids = None
        def __call__(self, _prompt, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        def decode(self, tokens, skip_special_tokens=False):
            return "tok"
    class Model:
        def __call__(self, **_kwargs):
            return types.SimpleNamespace(logits=torch.tensor([[[10.,9.,8.,7.,6.,5.,4.]]]), past_key_values=None, attentions=None, hidden_states=None)
    gen = DualStreamGenerator.__new__(DualStreamGenerator)
    gen.model_name = "fake"; gen.device = "cpu"; gen.tokenizer = Tok(); gen.model = Model()
    cfg = GenerationConfig(max_new_tokens=1, top_k=3, do_sample=False, compact_evidence=True, evidence_profile="DSA-CI-Standard", enable_heuristics=False, include_running_hash=False)
    result = gen.generate("x", cfg)
    assert len(result["frames"][0].topk) == 5
