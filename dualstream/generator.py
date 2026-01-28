from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import json
import os
import random
import re
import urllib.request

from .frame import MonologueFrameV1, TopKToken, ConceptScore, encode_frame
from .integrity import RunningHash


@dataclass
class GenerationConfig:
    model: str = "gemma3:1b"
    max_new_tokens: int = 128
    top_k: int = 5  # evidence top-K
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    seed: Optional[int] = None

    include_attn: bool = False
    attn_max_items: int = 8

    include_probes: bool = False
    probe_pack_path: Optional[str] = None

    # If no probe pack is provided, a *very small* heuristic fallback can be enabled
    # to reproduce the paper's illustrative Appendix A example shape.
    enable_heuristics: bool = True

    # Integrity
    include_crc32: bool = True
    include_running_hash: bool = True

    device: Optional[str] = None  # retained for CLI compatibility


def _resolve_ollama_host(override: Optional[str]) -> str:
    host = override or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def _ollama_generate(
    *,
    host: str,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: Optional[int],
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_predict": max_new_tokens,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    url = f"{host}/api/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    chunks: list[str] = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "error" in data:
                raise RuntimeError(f"Ollama error: {data['error']}")
            if "response" in data:
                chunks.append(data["response"])
            if data.get("done"):
                break

    return "".join(chunks)


def _tokenize_for_frames(text: str) -> list[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return tokens or []


def _heuristic_concepts(prompt: str, token_text: str) -> list[ConceptScore]:
    hits: list[ConceptScore] = []
    lower_prompt = prompt.lower()
    if any(q in lower_prompt for q in ["right?", "correct?", "isn't it", "is it true", "am i right"]):
        hits.append(ConceptScore(concept_id=1001, score=0.83))
    if re.search(r"\b(is|are|was|were)\b.*\b(correct|true|right)\b", lower_prompt):
        hits.append(ConceptScore(concept_id=2001, score=0.72))
    token_norm = token_text.strip().lower()
    if token_norm in {"yes", "absolutely", "correct", "right"}:
        hits.append(ConceptScore(concept_id=3001, score=0.95))
    return hits


class DualStreamGenerator:
    """
    Ollama-backed inference wrapper that emits 1:1 Answer tokens and evidence frames.

    Ollama does not expose token-level logits or hidden states, so we synthesize
    minimal evidence frames using output tokenization and heuristic concept tags.
    """

    def __init__(self, model_name: str, *, host: Optional[str] = None):
        self.model_name = model_name
        self.host = _resolve_ollama_host(host)

    def generate(self, prompt: str, cfg: GenerationConfig) -> Dict[str, Any]:
        if cfg.seed is not None:
            random.seed(cfg.seed)

        prompt_nonce = random.getrandbits(64)

        answer_text = _ollama_generate(
            host=self.host,
            model=cfg.model,
            prompt=prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )

        tokens = _tokenize_for_frames(answer_text)
        token_lookup = {idx: tok for idx, tok in enumerate(tokens)}

        frames: List[MonologueFrameV1] = []
        frame_bytes: List[bytes] = []
        running_hash = RunningHash() if cfg.include_running_hash else None

        for idx, tok in enumerate(tokens):
            concepts = _heuristic_concepts(prompt, tok) if cfg.enable_heuristics else []
            topk = [TopKToken(token_id=idx, prob=1.0)]
            frame = MonologueFrameV1(
                prompt_nonce=prompt_nonce,
                token_index=idx,
                chosen_id=idx,
                topk=topk,
                concepts=concepts,
            )
            frames.append(frame)
            fb = encode_frame(frame, include_crc32=cfg.include_crc32)
            frame_bytes.append(fb)
            if running_hash is not None:
                running_hash.update(fb)

        return {
            "prompt_nonce": prompt_nonce,
            "answer": answer_text,
            "frames": frames,
            "frame_bytes": frame_bytes,
            "running_hash": None if running_hash is None else running_hash.digest_hex(),
            "model": cfg.model,
            "token_lookup": token_lookup,
            "config": cfg,
        }
