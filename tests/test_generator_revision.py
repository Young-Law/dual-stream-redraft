from dualstream.generator import DualStreamGenerator


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = None


class _Model:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_generator_forwards_explicit_revision(monkeypatch):
    calls = {}

    class TokFactory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["tokenizer"] = (model_name, kwargs)
            return _Tokenizer()

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["model"] = (model_name, kwargs)
            return _Model()

    monkeypatch.setattr("dualstream.generator.AutoTokenizer", TokFactory)
    monkeypatch.setattr("dualstream.generator.AutoModelForCausalLM", ModelFactory)
    monkeypatch.setattr("dualstream.generator.torch.cuda.is_available", lambda: False)

    DualStreamGenerator(
        "org/model",
        revision="abc123",
        local_files_only=True,
        cache_dir="/tmp/cache",
    )

    assert calls["tokenizer"][1]["revision"] == "abc123"
    assert calls["model"][1]["revision"] == "abc123"


def test_generator_omits_revision_when_unset(monkeypatch):
    calls = {}

    class TokFactory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["tokenizer"] = kwargs
            return _Tokenizer()

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls["model"] = kwargs
            return _Model()

    monkeypatch.setattr("dualstream.generator.AutoTokenizer", TokFactory)
    monkeypatch.setattr("dualstream.generator.AutoModelForCausalLM", ModelFactory)
    monkeypatch.setattr("dualstream.generator.torch.cuda.is_available", lambda: False)

    DualStreamGenerator("org/model")

    assert "revision" not in calls["tokenizer"]
    assert "revision" not in calls["model"]
