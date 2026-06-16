import argparse
import json

import pytest

from dualstream import cli


class _FakeFrame:
    def __init__(self, token_index: int):
        self.token_index = token_index

    def to_dict(self):
        return {"token_index": self.token_index}


class _FakeTokenizer:
    def decode(self, _tokens):
        return "tok"


class _FakeGenerator:
    def __init__(self, *_args, **_kwargs):
        self.tokenizer = _FakeTokenizer()

    def generate(self, prompt, _cfg):
        return {
            "answer": f"answer::{prompt}",
            "frames": [_FakeFrame(0)],
            "frame_bytes": [b"frame"],
            "prompt_nonce": 123,
            "model": "fake-model",
            "running_hash": "abc",
        }


class _FakeFinding:
    def __init__(self, label: str):
        self.label = label


def test_generate_requires_exactly_one_prompt_source():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as both_exc:
        parser.parse_args(
            [
                "generate",
                "--prompt",
                "hello",
                "--prompt-file",
                "prompts.txt",
            ]
        )
    assert both_exc.value.code == 2

    with pytest.raises(SystemExit) as neither_exc:
        parser.parse_args(["generate"])
    assert neither_exc.value.code == 2


def test_generate_batch_writes_numbered_runs_and_manifest(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first\n\nsecond\n", encoding="utf-8")
    outdir = tmp_path / "runs"

    monkeypatch.setattr(cli, "DualStreamGenerator", _FakeGenerator)
    monkeypatch.setattr(cli, "render_monologue_text", lambda _frames, tokenizer_decode: tokenizer_decode([0]))
    monkeypatch.setattr(cli, "coherence_audit", lambda *_args, **_kwargs: [_FakeFinding("ok")])

    args = argparse.Namespace(
        model="fake",
        prompt=None,
        prompt_file=str(prompts),
        outdir=str(outdir),
        max_new_tokens=4,
        top_k=2,
        temperature=1.0,
        top_p=1.0,
        greedy=False,
        seed=None,
        include_attn=False,
        include_probes=False,
        probe_pack=None,
        no_heuristics=False,
        no_crc32=False,
        no_running_hash=False,
        device=None,
        offline=False,
        cache_dir=None,
    )

    rc = cli.cmd_generate(args)
    assert rc == 0

    run1 = outdir / "0001"
    run2 = outdir / "0002"
    for run in (run1, run2):
        assert (run / "answer.txt").exists()
        assert (run / "monologue.txt").exists()
        assert (run / "monologue.jsonl").exists()
        assert (run / "monologue.bin").exists()
        assert (run / "meta.json").exists()
        assert (run / "audit.json").exists()

    rows = [json.loads(line) for line in (outdir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["index"] for row in rows] == [1, 2]
    assert [row["prompt"] for row in rows] == ["first", "second"]
    assert rows[0]["run_dir"] == "0001"
    assert rows[1]["run_dir"] == "0002"
    assert rows[0]["answer_path"] == "0001/answer.txt"
    assert rows[1]["meta_path"] == "0002/meta.json"


def test_generate_prompt_file_manifest_includes_compact_evidence(tmp_path, monkeypatch):
    from dualstream.compact_evidence import encode_compact_sequence

    class _CompactFakeGenerator(_FakeGenerator):
        def generate(self, prompt, cfg):
            result = super().generate(prompt, cfg)
            result["compact_evidence_bytes"] = encode_compact_sequence([{"chosen_id": 1, "topk_ids": [1, 2, 3], "topk_scores": [.7, .2, .1]}])
            result["answer_token_ids"] = [1]
            return result

    prompts = tmp_path / "prompts.txt"
    prompts.write_text("first\nsecond\n", encoding="utf-8")
    outdir = tmp_path / "runs"
    monkeypatch.setattr(cli, "DualStreamGenerator", _CompactFakeGenerator)
    monkeypatch.setattr(cli, "render_monologue_text", lambda _frames, tokenizer_decode: tokenizer_decode([0]))
    monkeypatch.setattr(cli, "coherence_audit", lambda *_args, **_kwargs: [_FakeFinding("ok")])
    args = argparse.Namespace(
        model="fake", prompt=None, prompt_file=str(prompts), outdir=str(outdir), max_new_tokens=4,
        top_k=1, temperature=1.0, top_p=1.0, greedy=False, seed=None, include_attn=False,
        include_probes=False, probe_pack=None, no_heuristics=False, no_crc32=False, no_running_hash=False,
        device=None, offline=False, cache_dir=None, compact_evidence=True, evidence_profile="DSA-CI-Lite",
        adaptive_k=False, max_adaptive_k=None, chunk_token_capacity=256,
    )
    assert cli.cmd_generate(args) == 0
    rows = [json.loads(line) for line in (outdir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["compact_evidence_path"] == "0001/compact_evidence.dsae"
    assert (outdir / rows[1]["compact_evidence_path"]).exists()
    meta = json.loads((outdir / "0001" / "meta.json").read_text(encoding="utf-8"))
    assert meta["compact_evidence_token_count"] == 1
    assert meta["answer_token_count"] == 1


def test_generate_normalizes_compact_top_k_to_profile_base(tmp_path, monkeypatch):
    seen = {}
    class _CaptureGenerator(_FakeGenerator):
        def generate(self, prompt, cfg):
            seen["top_k"] = cfg.top_k
            return super().generate(prompt, cfg)
    monkeypatch.setattr(cli, "DualStreamGenerator", _CaptureGenerator)
    monkeypatch.setattr(cli, "render_monologue_text", lambda _frames, tokenizer_decode: tokenizer_decode([0]))
    monkeypatch.setattr(cli, "coherence_audit", lambda *_args, **_kwargs: [_FakeFinding("ok")])
    args = argparse.Namespace(
        model="fake", prompt="x", prompt_file=None, outdir=str(tmp_path), max_new_tokens=4,
        top_k=3, temperature=1.0, top_p=1.0, greedy=False, seed=None, include_attn=False,
        include_probes=False, probe_pack=None, no_heuristics=False, no_crc32=False, no_running_hash=False,
        device=None, offline=False, cache_dir=None, compact_evidence=True, evidence_profile="DSA-CI-Standard",
        adaptive_k=False, max_adaptive_k=None, chunk_token_capacity=256,
    )
    assert cli.cmd_generate(args) == 0
    assert seen["top_k"] == 5
