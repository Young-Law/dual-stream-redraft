from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from .compact_evidence import decode_compact_sequence, reconstruct_token_evidence, SCORE_TOLERANCE
from .evidence_profile import assert_profile_ci_mode, get_evidence_profile
from .retention import compute_evidence_budget_summary, assert_evidence_budget, assert_retention_floor


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    profile_id: str
    token_count: int
    elapsed_seconds: float
    peak_tracemalloc_bytes: int
    peak_rss_bytes: int
    raw_bytes_per_token: float
    adaptive_record_fraction: float
    max_effective_topk: int
    retention_floor_margin: float
    errors: list[str]

    def to_dict(self):
        return asdict(self)


def _rss_bytes() -> int:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss * 1024 if os.uname().sysname != "Darwin" else rss)
    except Exception:
        return 0


def find_compact_artifact(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        meta_path = p / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rel = meta.get("compact_evidence_path")
            if rel:
                q = p / rel
                if q.exists():
                    return q
        for name in ("compact_evidence.dsae", "compact_evidence.bin"):
            q = p / name
            if q.exists():
                return q
        raise FileNotFoundError(f"no compact evidence artifact found in {p}")
    return p


def _load_run_metadata(path: str | Path) -> dict[str, object]:
    p = Path(path)
    meta_path = p / "meta.json" if p.is_dir() else p.parent / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _enforce_metadata_binding(meta: dict[str, object], artifact_path: Path, decoded: dict[str, object], digest: str) -> None:
    if not meta:
        return
    expected_path = meta.get("compact_evidence_path")
    if expected_path and artifact_path.name != Path(str(expected_path)).name:
        raise ValueError("compact evidence path does not match run metadata")
    expected_sha = meta.get("compact_evidence_sha256")
    if expected_sha and str(expected_sha) != digest:
        raise ValueError("compact evidence sha256 does not match run metadata")
    expected_tokens = meta.get("compact_evidence_token_count")
    actual_tokens = int(decoded["header"].token_count)
    if expected_tokens is not None and int(expected_tokens) != actual_tokens:
        raise ValueError("compact evidence token count does not match run metadata")
    frame_tokens = meta.get("frame_token_count")
    if frame_tokens is not None and int(frame_tokens) != actual_tokens:
        raise ValueError("frame token count does not match compact evidence")
    answer_tokens = meta.get("answer_token_count")
    if answer_tokens is not None and int(answer_tokens) != actual_tokens:
        raise ValueError("answer token count does not match compact evidence")


def verify_evidence_artifact(path: str | Path, *, profile: str = "DSA-CI-Lite", ci_mode: str = "pr", enforce_budget: bool = True) -> VerificationReport:
    errors: list[str] = []
    start = time.perf_counter()
    tracemalloc.start()
    peak = 0
    raw_bpt = 0.0; floor_margin = 0.0; token_count = 0; adaptive_fraction = 0.0; max_eff = 0
    prof = get_evidence_profile(profile)
    try:
        assert_profile_ci_mode(prof, ci_mode)
        artifact_path = find_compact_artifact(path)
        meta = _load_run_metadata(path)
        data = artifact_path.read_bytes()
        decoded = decode_compact_sequence(data)
        _enforce_metadata_binding(meta, artifact_path, decoded, decoded["sha256"])
        if decoded["header"].profile_id != prof.profile_id.value:
            raise ValueError(f"artifact profile {decoded['header'].profile_id} does not match requested {prof.profile_id.value}")
        records = reconstruct_token_evidence(decoded)
        token_count = len(records)
        for i, rec in enumerate(records):
            if rec["token_index"] != i:
                raise ValueError("token indexes are not contiguous")
            if rec["effective_topk"] != len(rec["topk_ids"]) or len(rec["topk_ids"]) != len(rec["topk_scores"]):
                raise ValueError("top-k evidence shape mismatch")
            if rec["effective_topk"] < prof.base_k:
                raise ValueError("floor-starved token evidence")
            if any(score < -SCORE_TOLERANCE or score > 1 + SCORE_TOLERANCE for score in rec["topk_scores"]):
                raise ValueError("quantized score outside valid range")
        eff = [r["effective_topk"] for r in records]
        max_eff = max(eff, default=0)
        adaptive_fraction = sum(1 for k in eff if k > prof.base_k) / token_count if token_count else 0.0
        summary = compute_evidence_budget_summary(data, prof.profile_id.value)
        raw_bpt = summary.raw_bytes_per_token; floor_margin = summary.retention_floor_margin
        if enforce_budget:
            assert_evidence_budget(summary)
            assert_retention_floor(summary)
    except Exception as exc:
        errors.append(str(exc))
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    elapsed = time.perf_counter() - start
    rss = _rss_bytes()
    if enforce_budget:
        if elapsed > prof.verifier_time_seconds:
            errors.append(f"verification elapsed {elapsed:.6f}s exceeds profile budget {prof.verifier_time_seconds:.6f}s")
        budget_bytes = int(prof.verifier_peak_mib) * 1024 * 1024
        if peak > budget_bytes:
            errors.append(f"verification traced peak {peak} bytes exceeds profile budget {budget_bytes} bytes")
    return VerificationReport(not errors, prof.profile_id.value, token_count, elapsed, peak, rss, raw_bpt, adaptive_fraction, max_eff, floor_margin, errors)
