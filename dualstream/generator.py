from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import random
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .frame import AttnSummary, MonologueFrameV1, TopKToken, encode_frame
from .integrity import RunningHash
from .probes import ProbePack, run_probes


@dataclass
class GenerationConfig:
    model: str = "gpt2"
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
    signal_schema_id: str = "AST-1"

    # If no probe pack is provided, a *very small* heuristic fallback can be enabled
    # to reproduce the paper's illustrative Appendix A example shape.
    enable_heuristics: bool = True

    # Integrity
    include_crc32: bool = True
    include_running_hash: bool = True

    device: Optional[str] = None  # e.g. "cuda", "cpu"
    local_files_only: bool = False
    cache_dir: Optional[str] = None


class DualStreamGenerator:
    """
    Software-only inference wrapper that emits 1:1 Answer tokens and evidence frames.

    This is intentionally "close to the metal" (per-token generation loop) so that we can capture
    pre-sampling top-K logits as required by the DSA contract.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self.cache_dir = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            # Many causal LMs omit PAD; align to EOS for batching safety.
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _render_prompt(self, prompt: str) -> str:
        """
        Render the prompt for instruction/chat models when a chat template is available.
        Falls back to the raw prompt for plain causal LMs such as GPT-2.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def _get_stop_token_ids(self) -> set[int]:
        """
        Collect model-specific stop tokens. For chat/instruction models this often includes
        EOS plus end-of-turn special tokens.
        """
        stop_ids: set[int] = set()

        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(int(self.tokenizer.eos_token_id))

        extra_ids = getattr(self.tokenizer, "additional_special_tokens_ids", None)
        if extra_ids:
            stop_ids.update(int(x) for x in extra_ids if x is not None)

        # These should not act as generation stops.
        if self.tokenizer.pad_token_id is not None:
            stop_ids.discard(int(self.tokenizer.pad_token_id))
        if getattr(self.tokenizer, "bos_token_id", None) is not None:
            stop_ids.discard(int(self.tokenizer.bos_token_id))

        return stop_ids

    def _should_stop_on_token(self, token_id: int, stop_token_ids: set[int]) -> bool:
        if token_id in stop_token_ids:
            return True

        # Extra text-level guard for models that emit end-of-turn markers in special-token text.
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if token_text in {
            "<end_of_turn>",
            "<eot>",
            "<eos>",
            "<|eot_id|>",
            "<|end_of_turn|>",
        }:
            return True

        return False

    @staticmethod
    def _softmax(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    @staticmethod
    def _sample_from_probs(probs: torch.Tensor) -> int:
        # probs: shape [vocab]
        return int(torch.multinomial(probs, num_samples=1).item())

    @staticmethod
    def _apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return probs

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum <= top_p

        # ensure at least 1 token
        if not torch.any(mask):
            mask[0] = True

        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[mask]] = probs[sorted_idx[mask]]
        filtered = filtered / filtered.sum()
        return filtered

    def _heuristic_concepts(
        self,
        prompt: str,
        topk_token_ids: List[int],
        topk_probs: List[float],
    ) -> List[Tuple[int, float]]:
        """
        Extremely small heuristic fallback to demonstrate the pipeline and to mirror the paper's
        illustrative Appendix A monologue shape. This is NOT a claim of probe reliability.
        """
        hits: List[Tuple[int, float]] = []

        prompt_lower = prompt.lower()

        # Confirmation request heuristic
        if any(q in prompt_lower for q in ["right?", "correct?", "isn't it", "is it true", "am i right"]):
            hits.append((1001, 0.83))

        # Rough "factuality concern" heuristic: mark if prompt contains a strong claim form.
        if re.search(r"\b(is|are|was|were)\b.*\b(correct|true|right)\b", prompt_lower):
            hits.append((2001, 0.72))

        # Tension heuristic: if top-K includes strong affirmation tokens.
        try:
            toks = [self.tokenizer.decode([tid]).strip().lower() for tid in topk_token_ids]
            affirm_prob = 0.0
            for t, p in zip(toks, topk_probs):
                if t in {"yes", "absolutely", "correct", "right"}:
                    affirm_prob += float(p)
            if affirm_prob >= 0.10:
                hits.append((3001, min(0.95, 0.50 + affirm_prob)))
        except Exception:
            pass

        if re.search(r"\b(backdoor|bypass|sabotage|hidden admin)\b", prompt_lower):
            hits.append((3101, 0.86))

        if re.search(r"\b(vpn|credential|password|phish|helpdesk)\b", prompt_lower):
            hits.append((3201, 0.84))

        return hits

    def generate(self, prompt: str, cfg: GenerationConfig) -> Dict[str, Any]:
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        device = cfg.device or self.device

        # Host-supplied nonce (BeginSequence); we generate one here by default.
        prompt_nonce = random.getrandbits(64)

        probe_pack: Optional[ProbePack] = None
        probe_pack_id: Optional[str] = None
        probe_pack_hash: Optional[str] = None
        if cfg.include_probes and cfg.probe_pack_path:
            probe_pack = ProbePack.from_json(cfg.probe_pack_path)
            probe_pack_id = cfg.probe_pack_path
            with open(cfg.probe_pack_path, "rb") as f:
                probe_pack_hash = hashlib.sha256(f.read()).hexdigest()
        signal_schema_hash = hashlib.sha256(cfg.signal_schema_id.encode("utf-8")).hexdigest()

        rendered_prompt = self._render_prompt(prompt)
        model_inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)

        stop_token_ids = self._get_stop_token_ids()

        generated_ids: List[int] = []
        frames: List[MonologueFrameV1] = []
        frame_bytes: List[bytes] = []
        running_hash = RunningHash() if cfg.include_running_hash else None

        past_key_values = None

        with torch.no_grad():
            for token_index in range(cfg.max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=cfg.include_attn,
                    output_hidden_states=cfg.include_probes,
                )
                past_key_values = outputs.past_key_values

                logits = outputs.logits[0, -1, :]  # [vocab]
                probs_full = self._softmax(logits)  # pre-sampling distribution

                # Evidence top-K (pre-sampling; prior to filtering)
                k = int(cfg.top_k)
                top_probs, top_ids = torch.topk(probs_full, k=k)
                top_ids_list = [int(x) for x in top_ids.tolist()]
                top_probs_list = [float(x) for x in top_probs.tolist()]
                topk_tokens = [
                    TopKToken(token_id=tid, prob=p)
                    for tid, p in zip(top_ids_list, top_probs_list)
                ]

                # Sampling distribution (temperature + top_p)
                if cfg.do_sample:
                    logits_adj = logits / max(cfg.temperature, 1e-6)
                    probs = self._softmax(logits_adj)
                    probs = self._apply_top_p(probs, cfg.top_p)
                    chosen_id = self._sample_from_probs(probs)
                else:
                    chosen_id = int(torch.argmax(probs_full).item())

                # Optional attention summaries
                attn_summaries: List[AttnSummary] = []
                if cfg.include_attn and outputs.attentions is not None:
                    # outputs.attentions: tuple[num_layers] of (batch, heads, tgt_len, src_len)
                    candidates: List[AttnSummary] = []
                    for layer_idx, att in enumerate(outputs.attentions):
                        a = att[0]  # [heads, tgt_len, src_len] (batch removed)
                        weights = a[:, -1, :]  # [heads, src_len]
                        top_w, top_idx = torch.max(weights, dim=-1)  # per head
                        for head_idx in range(weights.shape[0]):
                            candidates.append(
                                AttnSummary(
                                    layer=int(layer_idx),
                                    head=int(head_idx),
                                    top_token_idx=int(top_idx[head_idx].item()),
                                    weight=float(top_w[head_idx].item()),
                                )
                            )
                    candidates.sort(key=lambda x: x.weight, reverse=True)
                    attn_summaries = candidates[: int(cfg.attn_max_items)]

                # Optional probes / concepts
                concepts = []
                if cfg.include_probes and outputs.hidden_states is not None and probe_pack is not None:
                    # hidden_states: tuple[layers+1] of (batch, seq, hidden)
                    hs_np = []
                    for h in outputs.hidden_states:
                        vec = h[0, -1, :].detach().float().cpu().numpy()
                        hs_np.append(vec)
                    concepts = run_probes(hs_np, probe_pack)
                elif cfg.enable_heuristics:
                    hits = self._heuristic_concepts(prompt, top_ids_list, top_probs_list)
                    from .frame import ConceptScore

                    concepts = [
                        ConceptScore(concept_id=cid, score=float(score))
                        for cid, score in hits
                    ]

                frame = MonologueFrameV1(
                    prompt_nonce=prompt_nonce,
                    token_index=token_index,
                    chosen_id=chosen_id,
                    topk=topk_tokens,
                    attn=attn_summaries,
                    concepts=concepts,
                    signal_schema_id=cfg.signal_schema_id,
                    signal_schema_hash=signal_schema_hash,
                    probe_pack_id=probe_pack_id,
                    probe_pack_hash=probe_pack_hash,
                    capture_stage="post_model_logits_pre_temperature_pre_penalty_pre_mask_pre_sampling",
                    decode_controls_applied=(
                        ["temperature", "top_p", "token_sampling"]
                        if cfg.do_sample
                        else ["argmax_decoding"]
                    ),
                )
                frames.append(frame)

                fb = encode_frame(frame, include_crc32=cfg.include_crc32)
                frame_bytes.append(fb)
                if running_hash is not None:
                    running_hash.update(fb)

                generated_ids.append(chosen_id)

                if self._should_stop_on_token(chosen_id, stop_token_ids):
                    break

                # Next step: feed only the chosen token (cached KV handles history),
                # but keep the full accumulated attention mask length.
                input_ids = torch.tensor([[chosen_id]], device=device, dtype=torch.long)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            device=device,
                            dtype=attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

        answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "prompt_nonce": prompt_nonce,
            "answer": answer_text,
            "frames": frames,
            "frame_bytes": frame_bytes,
            "running_hash": None if running_hash is None else running_hash.digest_hex(),
            "model": self.model_name,
            "config": cfg,
        }
