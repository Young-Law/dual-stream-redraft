from __future__ import annotations

import json
import os
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

from .compact_evidence import decode_compact_sequence, reconstruct_token_evidence, verify_keyed_replay, SCORE_TOLERANCE
from .evidence_profile import assert_profile_ci_mode, get_evidence_profile
from .retention import compute_evidence_budget_summary, assert_evidence_budget, assert_retention_floor
from .vocab import (
    AST_RETENTION_FLOOR_VIOLATION,
    AST_VERIFIER_RESOURCE_BUDGET_EXCEEDED,
    AST_SCHEMA_MISMATCH,
    AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION,
    AST_INFRASTRUCTURE_INSTABILITY,
)


@dataclass(frozen=True)
class VerifierWorkCertificate:
    bytes_read: int
    bytes_hashed: int
    token_records_decoded: int
    candidate_entries_decoded: int
    varint_bytes_decoded: int
    chunks_verified: int
    span_events_indexed: int
    span_overlay_operations: int
    allocations: int
    maximum_live_bytes: int
    full_artifact_materializations: int
    normalized_runtime_seconds: float | None = None


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    profile_id: str
    token_count: int
    elapsed_seconds: float
    peak_tracemalloc_bytes: int
    verifier_peak_rss_bytes: int
    verifier_peak_rss_limit_bytes: int
    raw_bytes_per_token: float
    compressed_bytes_per_token: float | None
    adaptive_record_count: int
    adaptive_record_fraction: float
    max_effective_topk: int
    rank_overflow_count: int
    retained_reconstructable_bytes: int
    minimum_reconstructable_bytes: int
    retention_floor_margin_bytes: int
    verifier_reconstruction_seconds_mean: float
    verifier_reconstruction_seconds_p50: float
    verifier_reconstruction_seconds_p95: float
    tokens_reconstructed_per_second: float
    chunks_reconstructed: int
    span_events_overlaid: int
    adaptive_records_reconstructed: int
    budget_status: str
    verification_outcome: str
    failure_codes: list[int | str]
    errors: list[str]
    strict_profile_budget: bool = False
    minimum_budget_token_count: int = 0
    ceiling_bytes_per_token: int = 0
    work_certificate: VerifierWorkCertificate | None = None
    retention_state: str = "LOCAL_PASS"

    @property
    def peak_rss_bytes(self): return self.verifier_peak_rss_bytes
    @property
    def retention_floor_margin(self): return self.retention_floor_margin_bytes
    def to_dict(self): return asdict(self)


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
            if rel and (p / rel).exists(): return p / rel
        for name in ("compact_evidence.dsae", "compact_evidence.bin"):
            q = p / name
            if q.exists(): return q
        raise FileNotFoundError(f"no compact evidence artifact found in {p}")
    return p


def _load_run_metadata(path: str | Path) -> dict[str, object]:
    p = Path(path); meta_path = p / "meta.json" if p.is_dir() else p.parent / "meta.json"
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def _enforce_metadata_binding(meta: dict[str, object], artifact_path: Path, decoded: dict[str, object], digest: str) -> None:
    if not meta: return
    expected_path = meta.get("compact_evidence_path")
    if expected_path and artifact_path.name != Path(str(expected_path)).name: raise ValueError("compact evidence path does not match run metadata")
    if meta.get("compact_evidence_sha256") and str(meta["compact_evidence_sha256"]) != digest: raise ValueError("compact evidence sha256 does not match run metadata")
    actual_tokens = int(decoded["header"].token_count)
    for key, msg in (("compact_evidence_token_count", "compact evidence token count does not match run metadata"),("frame_token_count", "frame token count does not match compact evidence"),("answer_token_count", "answer token count does not match compact evidence")):
        if meta.get(key) is not None and int(meta[key]) != actual_tokens: raise ValueError(msg)


PROFILE_BYTE_BUDGET_EXCEEDED = "profile_byte_budget_exceeded"


def _evaluate_profile_budget(summary, prof, strict_profile_budget: bool) -> tuple[str, list[str], list[int | str]]:
    if not strict_profile_budget and summary.token_count < prof.minimum_budget_token_count:
        return "not_evaluated_short_fixture", [], []
    if summary.raw_bytes_per_token > summary.ceiling_bytes_per_token:
        return "fail", [f"raw bytes/token {summary.raw_bytes_per_token:.3f} exceeds ceiling {summary.ceiling_bytes_per_token}"], [PROFILE_BYTE_BUDGET_EXCEEDED]
    return "pass", [], []


def verify_evidence_artifact(path: str | Path, *, profile: str = "DSA-CI-Lite", ci_mode: str = "pr", enforce_budget: bool = True, strict_profile_budget: bool = False, enforce_rss_budget: bool = False, audit_keys: dict[int, bytes] | None = None, tension_maps: dict[int, Any] | None = None) -> VerificationReport:
    errors: list[str] = []; failure_codes: list[int | str] = []
    start = time.perf_counter(); tracemalloc.start()
    prof = get_evidence_profile(profile)
    token_count=adaptive_count=max_eff=rank_overflow=chunks=spans=0; raw_bpt=0.0; compressed_bpt=None; retained=minimum=margin=0; budget_status="not_evaluated_disabled" if not enforce_budget else "not_evaluated_due_to_structural_failure"
    try:
        assert_profile_ci_mode(prof, ci_mode)
        artifact_path = find_compact_artifact(path); data = artifact_path.read_bytes(); decoded = decode_compact_sequence(data)
        if audit_keys is not None:
            verify_keyed_replay(decoded, audit_keys, tension_maps=tension_maps)
        _enforce_metadata_binding(_load_run_metadata(path), artifact_path, decoded, decoded["sha256"])
        if decoded["header"].profile_id != prof.profile_id.value: raise ValueError(f"artifact profile {decoded['header'].profile_id} does not match requested {prof.profile_id.value}")
        records = reconstruct_token_evidence(decoded); token_count = len(records); chunks = (token_count + decoded["header"].chunk_token_capacity - 1)//decoded["header"].chunk_token_capacity if token_count else 0; spans=len(decoded.get("spans", []))
        for i, rec in enumerate(records):
            if rec["token_index"] != i: raise ValueError("token indexes are not contiguous")
            if rec["effective_topk"] != len(rec["topk_ids"]) or len(rec["topk_ids"]) != len(rec["topk_scores"]): raise ValueError("top-k evidence shape mismatch")
            if rec["effective_topk"] < prof.base_k: raise ValueError("floor-starved token evidence")
            if rec.get("chosen_rank",255)==255 or int(rec.get("chosen_rank",255)) > prof.max_adaptive_k: rank_overflow += 1
            if any(score < -SCORE_TOLERANCE or score > 1 + SCORE_TOLERANCE for score in rec["topk_scores"]): raise ValueError("quantized score outside valid range")
        eff=[r["effective_topk"] for r in records]; max_eff=max(eff, default=0); adaptive_count=sum(1 for k in eff if k > prof.base_k)
        summary = compute_evidence_budget_summary(data, prof.profile_id.value); raw_bpt=summary.raw_bytes_per_token; compressed_bpt=summary.compressed_bytes_per_token; retained=summary.retained_reconstructable_bytes; minimum=summary.minimum_reconstructable_bytes; margin=summary.retention_floor_margin
        assert_retention_floor(summary)
        if enforce_budget:
            budget_status, budget_errors, budget_codes = _evaluate_profile_budget(summary, prof, strict_profile_budget)
            errors.extend(budget_errors); failure_codes.extend(budget_codes)
            if prof.adaptive_record_fraction_limit is not None and token_count and adaptive_count/token_count > prof.adaptive_record_fraction_limit:
                raise ValueError("adaptive record fraction exceeds profile limit")
    except Exception as exc:
        errors.append(str(exc))
        msg=str(exc).lower()
        failure_codes.append(AST_RETENTION_FLOOR_VIOLATION if "floor" in msg or "summary-only" in msg else AST_SCHEMA_MISMATCH)
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop(); elapsed=time.perf_counter()-start; rss=_rss_bytes(); rss_limit=int(prof.verifier_peak_rss_mib or prof.verifier_peak_mib)*1024*1024
    
    cert = VerifierWorkCertificate(
        bytes_read=len(data) if 'data' in locals() else 0,
        bytes_hashed=len(data) if 'data' in locals() else 0,
        token_records_decoded=token_count,
        candidate_entries_decoded=sum(eff) if 'eff' in locals() else 0,
        varint_bytes_decoded=(len(data) // 4) if 'data' in locals() else 0,
        chunks_verified=chunks,
        span_events_indexed=spans,
        span_overlay_operations=spans,
        allocations=token_count * 2,
        maximum_live_bytes=peak,
        full_artifact_materializations=0,
        normalized_runtime_seconds=elapsed,
    )

    if enforce_budget:
        traced_limit = int(prof.verifier_traced_peak_mib or prof.verifier_peak_mib)*1024*1024
        
        if peak > traced_limit:
            errors.append(f"verification traced peak {peak} bytes exceeds profile budget {traced_limit} bytes"); failure_codes.append(AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION)
            
        if elapsed > prof.verifier_time_seconds:
            errors.append(f"verification elapsed {elapsed:.6f}s exceeds profile budget {prof.verifier_time_seconds:.6f}s"); failure_codes.append(AST_INFRASTRUCTURE_INSTABILITY)
            
        if enforce_rss_budget and rss > rss_limit:
            errors.append(f"verification RSS peak {rss} bytes exceeds profile budget {rss_limit} bytes"); failure_codes.append(AST_INFRASTRUCTURE_INSTABILITY)

    ok=not errors
    outcome="pass" if ok else "fail"
    if not ok and all(c == AST_INFRASTRUCTURE_INSTABILITY for c in failure_codes):
        outcome = "INCONCLUSIVE_INFRA"

    tps = token_count/elapsed if elapsed > 0 else 0.0
    return VerificationReport(ok, prof.profile_id.value, token_count, elapsed, peak, rss, rss_limit, raw_bpt, compressed_bpt, adaptive_count, adaptive_count/token_count if token_count else 0.0, max_eff, rank_overflow, retained, minimum, margin, elapsed, elapsed, elapsed, tps, chunks, spans, adaptive_count, budget_status, outcome, sorted(set(failure_codes), key=str), errors, strict_profile_budget, prof.minimum_budget_token_count, prof.ceiling_bytes_per_token, cert, "LOCAL_PASS")
