from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_read_json(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def summarize_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    if not meta:
        return {}
    cfg = meta.get("config", {})
    return {
        "model": meta.get("model") or cfg.get("model"),
        "prompt_nonce": meta.get("prompt_nonce"),
        "running_hash": meta.get("running_hash"),
        "sampling": {
            "max_new_tokens": cfg.get("max_new_tokens"),
            "top_k": cfg.get("top_k"),
            "temperature": cfg.get("temperature"),
            "top_p": cfg.get("top_p"),
            "do_sample": cfg.get("do_sample"),
        },
        "features": {
            "include_attn": cfg.get("include_attn"),
            "include_probes": cfg.get("include_probes"),
            "probe_pack_path": cfg.get("probe_pack_path"),
        },
    }


def summarize_audit(audit: Any) -> dict[str, Any]:
    if not audit:
        return {}
    if isinstance(audit, dict):
        if "summary" in audit:
            return audit["summary"]
        return audit
    if isinstance(audit, list):
        findings_by_kind: dict[str, int] = {}
        for row in audit:
            kind = str(row.get("kind", "unknown"))
            findings_by_kind[kind] = findings_by_kind.get(kind, 0) + 1
        return {
            "num_findings": len(audit),
            "counts_by_kind": findings_by_kind,
        }
    return {"value": str(audit)}


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"num_frames": 0}
    with_topk = sum(1 for fr in frames if fr.get("topk_tokens"))
    with_concepts = sum(1 for fr in frames if fr.get("concepts"))
    with_attention = sum(1 for fr in frames if fr.get("attn"))
    return {
        "num_frames": len(frames),
        "with_topk": with_topk,
        "with_concepts": with_concepts,
        "with_attention": with_attention,
    }


def discover_artifacts(output_dir: str | Path) -> list[dict[str, Any]]:
    out = Path(output_dir)
    if not out.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for p in sorted(out.rglob("*")):
        if not p.is_file():
            continue
        artifacts.append(
            {
                "name": p.name,
                "relative_path": str(p.relative_to(out)),
                "size": p.stat().st_size,
            }
        )
    return artifacts


def load_generation_artifacts(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    answer_path = out / "answer.txt"
    monologue_path = out / "monologue.txt"
    frames_path = out / "monologue.jsonl"
    meta_path = out / "meta.json"
    audit_path = out / "audit.json"

    frames = read_jsonl(frames_path)
    meta = _safe_read_json(meta_path)
    audit = _safe_read_json(audit_path)

    return {
        "answer_text": answer_path.read_text(encoding="utf-8") if answer_path.exists() else "",
        "monologue_text": monologue_path.read_text(encoding="utf-8") if monologue_path.exists() else "",
        "frames": frames,
        "frames_summary": summarize_frames(frames),
        "meta": meta,
        "meta_summary": summarize_meta(meta),
        "audit": audit,
        "audit_summary": summarize_audit(audit),
        "artifacts": discover_artifacts(out),
    }


def load_arc_artifacts(output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    predictions = _safe_read_json(out / "predictions.json")
    metrics = _safe_read_json(out / "summary_metrics.json")
    rankings = _safe_read_json(out / "candidate_rankings.json")
    audit = _safe_read_json(out / "audit.json")
    trace = read_jsonl(out / "trace.jsonl")

    return {
        "predictions": predictions,
        "metrics": metrics,
        "candidate_rankings": rankings,
        "audit": audit,
        "audit_summary": summarize_audit(audit),
        "trace": trace,
        "trace_summary": summarize_frames(trace),
        "artifacts": discover_artifacts(out),
    }
