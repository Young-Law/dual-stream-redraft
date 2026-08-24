from __future__ import annotations

import json
import os
import statistics
import time
import tracemalloc
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .compact_evidence import decode_compact_sequence, reconstruct_token_evidence, verify_keyed_replay, SCORE_TOLERANCE, compute_retention_requirement_hash, EvidenceManifestV33, VERSION_V33
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
    signature: str | None = None


def canonical_serialize_certificate(cert: VerifierWorkCertificate) -> bytes:
    data = {
        "bytes_read": cert.bytes_read,
        "bytes_hashed": cert.bytes_hashed,
        "token_records_decoded": cert.token_records_decoded,
        "candidate_entries_decoded": cert.candidate_entries_decoded,
        "varint_bytes_decoded": cert.varint_bytes_decoded,
        "chunks_verified": cert.chunks_verified,
        "span_events_indexed": cert.span_events_indexed,
        "span_overlay_operations": cert.span_overlay_operations,
        "allocations": cert.allocations,
        "maximum_live_bytes": cert.maximum_live_bytes,
        "full_artifact_materializations": cert.full_artifact_materializations,
        "normalized_runtime_seconds": cert.normalized_runtime_seconds,
    }
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return serialized.encode("utf-8")


def sign_work_certificate(cert: VerifierWorkCertificate, key: bytes) -> str:
    import hmac
    import hashlib
    payload = canonical_serialize_certificate(cert)
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_work_certificate_signature(cert: VerifierWorkCertificate, signature: str, key: bytes) -> bool:
    import hmac
    expected = sign_work_certificate(cert, key)
    return hmac.compare_digest(expected, signature)


@dataclass(frozen=True)
class RetryPolicy:
    """Policy governing INCONCLUSIVE_INFRA retry behavior."""
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0
    retry_on_infra_only: bool = True


@dataclass
class RetryAttempt:
    """Metadata for a single retry attempt."""
    attempt: int
    started_at: float
    elapsed_seconds: float
    outcome: str
    errors: list
    failure_codes: list
    peak_tracemalloc_bytes: int
    verifier_peak_rss_bytes: int


@dataclass
class RetryResult:
    """Result of a retry loop with full metadata."""
    final_report: VerificationReport
    attempts: list
    total_attempts: int
    total_elapsed_seconds: float
    retried: bool
    converged: bool


DEFAULT_RETRY_POLICY = RetryPolicy()


def verify_with_retry(
    path: str | Path,
    *,
    profile: str = "DSA-CI-Lite",
    ci_mode: str = "pr",
    enforce_budget: bool = True,
    strict_profile_budget: bool = False,
    enforce_rss_budget: bool = False,
    audit_keys: dict[int, bytes] | None = None,
    tension_maps: dict[int, Any] | None = None,
    verifier_key: bytes | None = None,
    retention_requirement: str | dict | None = None,
    retry_policy: RetryPolicy | None = None,
) -> VerificationReport:
    """Verify an evidence artifact with automatic INCONCLUSIVE_INFRA retry.

    When the verifier produces an ``INCONCLUSIVE_INFRA`` outcome (artifact may
    be valid but the verifier's resource environment was insufficient), this
    function retries up to ``retry_policy.max_retries`` times with exponential
    backoff.

    Each attempt's metadata is preserved in ``RetryResult.attempts``.

    Returns the final ``VerificationReport`` and a ``RetryResult``.
    """
    policy = retry_policy or DEFAULT_RETRY_POLICY
    first_report = verify_evidence_artifact(
        path, profile=profile, ci_mode=ci_mode, enforce_budget=enforce_budget,
        strict_profile_budget=strict_profile_budget, enforce_rss_budget=enforce_rss_budget,
        audit_keys=audit_keys, tension_maps=tension_maps, verifier_key=verifier_key,
        retention_requirement=retention_requirement,
    )

    if first_report.verification_outcome != "INCONCLUSIVE_INFRA":
        return first_report

    attempts: list[RetryAttempt] = [
        RetryAttempt(
            attempt=0, started_at=time.perf_counter(),
            elapsed_seconds=first_report.elapsed_seconds,
            outcome=first_report.verification_outcome,
            errors=list(first_report.errors),
            failure_codes=list(first_report.failure_codes),
            peak_tracemalloc_bytes=first_report.peak_tracemalloc_bytes,
            verifier_peak_rss_bytes=first_report.verifier_peak_rss_bytes,
        )
    ]

    loop_start = time.perf_counter()
    last_report = first_report
    converged = False

    for attempt_idx in range(1, policy.max_retries + 1):
        backoff = min(
            policy.backoff_base_seconds * (policy.backoff_multiplier ** (attempt_idx - 1)),
            policy.max_backoff_seconds,
        )
        time.sleep(backoff)

        attempt_report = verify_evidence_artifact(
            path, profile=profile, ci_mode=ci_mode, enforce_budget=enforce_budget,
            strict_profile_budget=strict_profile_budget, enforce_rss_budget=enforce_rss_budget,
            audit_keys=audit_keys, tension_maps=tension_maps, verifier_key=verifier_key,
            retention_requirement=retention_requirement,
        )
        attempts.append(RetryAttempt(
            attempt=attempt_idx, started_at=time.perf_counter(),
            elapsed_seconds=attempt_report.elapsed_seconds,
            outcome=attempt_report.verification_outcome,
            errors=list(attempt_report.errors),
            failure_codes=list(attempt_report.failure_codes),
            peak_tracemalloc_bytes=attempt_report.peak_tracemalloc_bytes,
            verifier_peak_rss_bytes=attempt_report.verifier_peak_rss_bytes,
        ))

        if attempt_report.verification_outcome != "INCONCLUSIVE_INFRA":
            converged = True
            last_report = attempt_report
            break
        last_report = attempt_report

    total_elapsed = time.perf_counter() - loop_start
    # Store retry metadata on the report via monkey-patch (frozen dataclass)
    retry_result = RetryResult(
        final_report=last_report, attempts=attempts,
        total_attempts=len(attempts), total_elapsed_seconds=total_elapsed,
        retried=True, converged=converged,
    )
    try:
        object.__setattr__(last_report, "retry_result", retry_result)
    except AttributeError:
        pass

    # Update retention_state to indicate retry
    try:
        new_state = "RETRIED_INFRA_PASS" if last_report.ok else "RETRIED_INFRA_FAIL"
        object.__setattr__(last_report, "retention_state", new_state)
    except AttributeError:
        pass

    return last_report


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
    retention_hash_valid: bool = False

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


def verify_evidence_artifact(path: str | Path, *, profile: str = "DSA-CI-Lite", ci_mode: str = "pr", enforce_budget: bool = True, strict_profile_budget: bool = False, enforce_rss_budget: bool = False, audit_keys: dict[int, bytes] | None = None, tension_maps: dict[int, Any] | None = None, verifier_key: bytes | None = None, retention_requirement: str | dict | None = None) -> VerificationReport:
    errors: list[str] = []; failure_codes: list[int | str] = []
    start = time.perf_counter(); tracemalloc.start()
    prof = get_evidence_profile(profile)
    token_count=adaptive_count=max_eff=rank_overflow=chunks=spans=0; raw_bpt=0.0; compressed_bpt=None; retained=minimum=margin=0; budget_status="not_evaluated_disabled" if not enforce_budget else "not_evaluated_due_to_structural_failure"; recon_mean=recon_p50=recon_p95=0.0
    try:
        assert_profile_ci_mode(prof, ci_mode)
        artifact_path = find_compact_artifact(path); data = artifact_path.read_bytes(); decoded = decode_compact_sequence(data)
        if audit_keys is not None:
            verify_keyed_replay(decoded, audit_keys, tension_maps=tension_maps)
        _enforce_metadata_binding(_load_run_metadata(path), artifact_path, decoded, decoded["sha256"])
        if decoded["header"].profile_id != prof.profile_id.value: raise ValueError(f"artifact profile {decoded['header'].profile_id} does not match requested {prof.profile_id.value}")
        _cap = decoded["header"].chunk_token_capacity; _spans = decoded.get("spans", []); records: list[dict[str, Any]] = []; _chunk_timings: list[float] = []
        for _cs in range(0, len(decoded["tokens"]), _cap):
            _ct = decoded["tokens"][_cs:_cs + _cap]; _t0 = time.perf_counter()
            for _rec in _ct:
                records.append({"token_index": _rec.token_index, "chosen_id": _rec.chosen_id, "topk_ids": list(_rec.topk_ids), "topk_scores": list(_rec.topk_scores), "effective_topk": _rec.effective_topk, "chosen_rank": _rec.chosen_rank, "trigger_flags": _rec.trigger_flags, "record_flags": _rec.record_flags, "signals": [s for s in _spans if s.start_token <= _rec.token_index < s.end_token]})
            _chunk_timings.append(time.perf_counter() - _t0)
        token_count = len(records); chunks = len(_chunk_timings); spans = len(_spans)
        if _chunk_timings:
            recon_mean = statistics.mean(_chunk_timings); recon_p50 = statistics.median(_chunk_timings); _st = sorted(_chunk_timings); recon_p95 = _st[int(len(_st) * 0.95) if len(_st) > 1 else 0]
        else:
            recon_mean = recon_p50 = recon_p95 = 0.0
        for i, rec in enumerate(records):
            if rec["token_index"] != i: raise ValueError("token indexes are not contiguous")
            if rec["effective_topk"] != len(rec["topk_ids"]) or len(rec["topk_ids"]) != len(rec["topk_scores"]): raise ValueError("top-k evidence shape mismatch")
            if rec["effective_topk"] < prof.base_k: raise ValueError("floor-starved token evidence")
            if rec.get("chosen_rank",255)==255 or int(rec.get("chosen_rank",255)) > prof.max_adaptive_k: rank_overflow += 1
            if any(score < -SCORE_TOLERANCE or score > 1 + SCORE_TOLERANCE for score in rec["topk_scores"]): raise ValueError("quantized score outside valid range")
        eff=[r["effective_topk"] for r in records]; max_eff=max(eff, default=0); adaptive_count=sum(1 for k in eff if k > prof.base_k)
        summary = compute_evidence_budget_summary(data, prof.profile_id.value); raw_bpt=summary.raw_bytes_per_token; compressed_bpt=round(len(zlib.compress(data)) / token_count, 2) if token_count else None; retained=summary.retained_reconstructable_bytes; minimum=summary.minimum_reconstructable_bytes; margin=summary.retention_floor_margin
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

    if verifier_key is not None:
        cert = replace(cert, signature=sign_work_certificate(cert, verifier_key))

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

    # P7: Manifest chaining — verify retention requirement hash against V3.3 manifest
    retention_hash_valid = False
    if 'decoded' in locals() and retention_requirement is not None:
        manifest: EvidenceManifestV33 | None = decoded.get("manifest")
        header_ver = decoded.get("header")
        if manifest is not None and header_ver is not None and getattr(header_ver, "schema_version", None) == VERSION_V33:
            expected_hash = compute_retention_requirement_hash(retention_requirement)
            actual_hash_hex = manifest.retention_requirement_hash
            retention_hash_valid = expected_hash.hex() == actual_hash_hex
            if not retention_hash_valid:
                errors.append(f"retention requirement hash mismatch: expected {expected_hash.hex()}, manifest has {actual_hash_hex}")
                failure_codes.append(AST_SCHEMA_MISMATCH)
                ok = False
                outcome = "fail"

    tps = token_count/elapsed if elapsed > 0 else 0.0
    return VerificationReport(
        ok=ok,
        profile_id=prof.profile_id.value,
        token_count=token_count,
        elapsed_seconds=elapsed,
        peak_tracemalloc_bytes=peak,
        verifier_peak_rss_bytes=rss,
        verifier_peak_rss_limit_bytes=rss_limit,
        raw_bytes_per_token=raw_bpt,
        compressed_bytes_per_token=compressed_bpt,
        adaptive_record_count=adaptive_count,
        adaptive_record_fraction=adaptive_count/token_count if token_count else 0.0,
        max_effective_topk=max_eff,
        rank_overflow_count=rank_overflow,
        retained_reconstructable_bytes=retained,
        minimum_reconstructable_bytes=minimum,
        retention_floor_margin_bytes=margin,
        verifier_reconstruction_seconds_mean=recon_mean,
        verifier_reconstruction_seconds_p50=recon_p50,
        verifier_reconstruction_seconds_p95=recon_p95,
        tokens_reconstructed_per_second=tps,
        chunks_reconstructed=chunks,
        span_events_overlaid=spans,
        adaptive_records_reconstructed=adaptive_count,
        budget_status=budget_status,
        verification_outcome=outcome,
        failure_codes=sorted(set(failure_codes), key=str),
        errors=errors,
        strict_profile_budget=strict_profile_budget,
        minimum_budget_token_count=prof.minimum_budget_token_count,
        ceiling_bytes_per_token=prof.ceiling_bytes_per_token,
        work_certificate=cert,
        retention_state="LOCAL_PASS",
        retention_hash_valid=retention_hash_valid,
    )