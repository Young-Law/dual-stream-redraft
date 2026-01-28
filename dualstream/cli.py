from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from .audit import coherence_audit
from .frame import MonologueFrameV1, TopKToken, encode_frame
from .integrity import RunningHash
from .render import render_monologue_text


@dataclass
class GenerationConfig:
    model: str = "llama3"
    max_new_tokens: int = 128
    top_k: int = 5
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    seed: Optional[int] = None

    include_attn: bool = False
    include_probes: bool = False
    probe_pack_path: Optional[str] = None
    enable_heuristics: bool = True

    include_crc32: bool = True
    include_running_hash: bool = True


def _ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: Optional[int],
    do_sample: bool,
    timeout_s: float,
) -> str:
    url = host.rstrip("/") + "/api/generate"
    options: Dict[str, object] = {
        "num_predict": int(max_new_tokens),
        "top_k": int(top_k),
        "top_p": float(top_p),
    }
    if seed is not None:
        options["seed"] = int(seed)
    if do_sample:
        options["temperature"] = float(temperature)
    else:
        options["temperature"] = 0.0

    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
    ).encode("utf-8")
    req = urlrequest.Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urlerror.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON response.") from exc
    if "response" not in data:
        raise RuntimeError(f"Ollama response missing 'response' field: {data}")
    return str(data["response"])


def _tokenize_response(text: str) -> Tuple[List[str], Dict[str, int]]:
    tokens = re.findall(r"\S+|\s+", text)
    token_to_id: Dict[str, int] = {}
    for tok in tokens:
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id) + 1
    return tokens, token_to_id


def _build_frames(
    tokens: List[str],
    token_to_id: Dict[str, int],
    *,
    prompt_nonce: int,
    top_k: int,
    include_crc32: bool,
    include_running_hash: bool,
) -> Tuple[List[MonologueFrameV1], List[bytes], Optional[str]]:
    frames: List[MonologueFrameV1] = []
    frame_bytes: List[bytes] = []
    running_hash = RunningHash() if include_running_hash else None
    top_k = max(1, int(top_k))
    for token_index, tok in enumerate(tokens):
        chosen_id = token_to_id[tok]
        topk_tokens = [TopKToken(token_id=chosen_id, prob=1.0)]
        if top_k > 1:
            topk_tokens.extend([TopKToken(token_id=0, prob=0.0)] * (top_k - 1))
        frame = MonologueFrameV1(
            prompt_nonce=prompt_nonce,
            token_index=token_index,
            chosen_id=chosen_id,
            topk=topk_tokens,
            attn=[],
            concepts=[],
        )
        frames.append(frame)
        fb = encode_frame(frame, include_crc32=include_crc32)
        frame_bytes.append(fb)
        if running_hash is not None:
            running_hash.update(fb)
    return frames, frame_bytes, None if running_hash is None else running_hash.digest_hex()


def _decode_token(token_to_id: Dict[str, int]) -> Callable[[int], str]:
    id_to_token = {tid: tok for tok, tid in token_to_id.items()}

    def decode(tid: int) -> str:
        return id_to_token.get(int(tid), "<UNK>")

    return decode


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = GenerationConfig(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
        seed=args.seed,
        include_attn=args.include_attn,
        include_probes=args.include_probes,
        probe_pack_path=args.probe_pack,
        enable_heuristics=not args.no_heuristics,
        include_crc32=not args.no_crc32,
        include_running_hash=not args.no_running_hash,
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    answer_text = _ollama_generate(
        host=args.ollama_host,
        model=cfg.model,
        prompt=args.prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        seed=cfg.seed,
        do_sample=cfg.do_sample,
        timeout_s=args.ollama_timeout,
    )
    tokens, token_to_id = _tokenize_response(answer_text)
    prompt_nonce = random.getrandbits(64)
    frames, frame_bytes, running_hash = _build_frames(
        tokens,
        token_to_id,
        prompt_nonce=prompt_nonce,
        top_k=cfg.top_k,
        include_crc32=cfg.include_crc32,
        include_running_hash=cfg.include_running_hash,
    )
    decoder = _decode_token(token_to_id)

    monologue_text = render_monologue_text(frames, tokenizer_decode=decoder)

    (outdir / "answer.txt").write_text(answer_text, encoding="utf-8")
    (outdir / "monologue.txt").write_text(monologue_text, encoding="utf-8")

    # JSONL evidence frames
    with (outdir / "monologue.jsonl").open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr.to_dict(), ensure_ascii=False) + "\n")

    # Raw binary stream (concatenated frames) for low-level consumers
    with (outdir / "monologue.bin").open("wb") as f:
        for fb in frame_bytes:
            f.write(fb)

    meta = {
        "prompt_nonce": prompt_nonce,
        "model": cfg.model,
        "backend": "ollama",
        "ollama_host": args.ollama_host,
        "running_hash": running_hash,
        "config": {
            "model": cfg.model,
            "max_new_tokens": cfg.max_new_tokens,
            "top_k": cfg.top_k,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "do_sample": cfg.do_sample,
            "include_attn": cfg.include_attn,
            "include_probes": cfg.include_probes,
            "probe_pack_path": cfg.probe_pack_path,
        },
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Optional coherence audit
    findings = coherence_audit(answer_text, frames, decode_token=decoder)
    (outdir / "audit.json").write_text(
        json.dumps([f.__dict__ for f in findings], indent=2),
        encoding="utf-8",
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dualstream", description="Dual-Stream Architecture reference implementation")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate answer + monologue evidence for a prompt")
    g.add_argument("--model", default="llama3", help="Ollama model name")
    g.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    g.add_argument("--ollama-timeout", type=float, default=60.0, help="Ollama request timeout (seconds)")
    g.add_argument("--prompt", required=True, help="User prompt")
    g.add_argument("--outdir", default=".", help="Output directory")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--top-k", type=int, default=5, help="Top-K evidence tokens per step (pre-sampling)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--greedy", action="store_true", help="Disable sampling (argmax)")
    g.add_argument("--seed", type=int, default=None)

    g.add_argument("--include-attn", action="store_true", help="Emit attention summaries (slow)")
    g.add_argument("--include-probes", action="store_true", help="Run concept probes (requires hidden states; slow)")
    g.add_argument("--probe-pack", default=None, help="Path to probe pack JSON")
    g.add_argument("--no-heuristics", action="store_true", help="Disable heuristic concepts")

    g.add_argument("--no-crc32", action="store_true", help="Do not append crc32 to frames")
    g.add_argument("--no-running-hash", action="store_true", help="Disable running hash accumulation")

    g.set_defaults(func=cmd_generate)
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
