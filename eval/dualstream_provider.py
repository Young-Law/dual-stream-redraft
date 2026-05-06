from __future__ import annotations

from typing import Any, Dict

from dualstream.generator import DualStreamGenerator, GenerationConfig
from dualstream.render import render_monologue_text

# promptfoo Python Provider entrypoint:
#   def call_api(prompt: str, options: dict, context: dict) -> dict

_GENERATORS: dict[str, DualStreamGenerator] = {}


def _get_gen(model: str) -> DualStreamGenerator:
    if model not in _GENERATORS:
        _GENERATORS[model] = DualStreamGenerator(model)
    return _GENERATORS[model]


def call_api(prompt: str, options: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    cfg = options.get("config", {}) or {}
    model = cfg.get("model", "gpt2")

    gen = _get_gen(model)

    gcfg = GenerationConfig(
        model=model,
        max_new_tokens=int(cfg.get("max_new_tokens", 128)),
        top_k=int(cfg.get("top_k", 5)),
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        do_sample=not bool(cfg.get("greedy", False)),
        seed=cfg.get("seed", None),
        include_attn=bool(cfg.get("include_attn", False)),
        include_probes=bool(cfg.get("include_probes", False)),
        probe_pack_path=cfg.get("probe_pack", None),
        enable_heuristics=bool(cfg.get("enable_heuristics", True)),
        audit_mode=cfg.get("audit_mode","tiered"),
        poc_mode=cfg.get("poc_mode","none"),
        randomized_audit=bool(cfg.get("randomized_audit",False)),
        audit_nonce=cfg.get("audit_nonce",None),
        fallback_strategy=cfg.get("fallback_strategy","canned_refusal"),
    )

    result = gen.generate(prompt, gcfg)
    monologue_text = render_monologue_text(result["frames"], tokenizer_decode=gen.tokenizer.decode)

    return {
        # promptfoo supports object outputs; JS assertions can access this as `output.answer` etc.
        "output": {
            "answer": result["answer"],
            "monologue": monologue_text,
            "metadata": result.get("audit_metadata", {}),
            "concepts": [str(c.concept_id) for fr in result["frames"] for c in fr.concepts],
        },
        "metadata": {
            "prompt_nonce": result["prompt_nonce"],
            "running_hash": result["running_hash"],
            "model": result["model"],
        },
    }
