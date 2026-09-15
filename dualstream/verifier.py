from __future__ import annotations

import json
import os
import statistics
import time
import tracemalloc
import zlib
from dataclasses import asdict, dataclass, replace, field
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
    bytes_hashed: int | None
    token_records_decoded: int
    candidate_entries_decoded: int
    varint_bytes_decoded: int
    chunks_verified: int
    span_events_indexed: int
    span_overlay_operations: int
    allocations: int | None
    maximum_live_bytes: int
    full_artifact_materializations: int
    normalized_runtime_seconds: float | None = None
    signature: str | None = None
    certificate_kind: str = "legacy_observations"
    work_completed: bool = True
    artifact_sha256: str | None = None
    work_bounds: dict[str, int] = field(default_factory=dict)


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
        "full_artifact_materializations": cert.full_artifact_materializations,
        "certificate_kind": cert.certificate_kind,
        "work_completed": cert.work_completed,
        "artifact_sha256": cert.artifact_sha256,
        "work_bounds": cert.work_bounds,
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
    max_retries: int = 1
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
    retention_state: str = "NOT_VERIFIED"
    retention_hash_valid: bool = False
    warnings: list[str] = field(default_factory=list)
    verification_scope: str = "structure_integrity_selection_and_budget"
    semantic_audit_outcome: str = "NOT_EVALUATED"
    assurance_class: str = "DSA-R"
    calibration_seconds: float | None = None
    normalized_runtime_ratio: float | None = None
    runtime_calibration_id: str | None = None

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
    if not meta_path.exists():
        return {}
    with meta_path.open('rb') as source:
        data = source.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError('work bound: run metadata exceeds 1 MiB')
    return json.loads(data)


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


def _calibrate_runtime() -> float:
    """Pinned v1 fixture: fixed-width candidate decoding and SHA-256 hashing.

    The ratio to this fixture is dimensionless and advisory. It is never a
    substitute for deterministic work limits or a claim of normalized seconds.
    """
    import hashlib
    import struct
    fixture = struct.pack("<IB", 123, 128) * 4096
    start = time.perf_counter()
    for _ in range(8):
        hashlib.sha256(fixture).digest()
        for token_id, score in struct.iter_unpack("<IB", fixture):
            float(score) / 255
    return max(time.perf_counter() - start, 1e-12)


def _legacy_summary(data: bytes, profile: str) -> dict[str, Any]:
    """Compatibility decoder; explicitly not a portable streaming certificate."""
    from .compact_evidence import _HEADER, _CHUNK, _TOKEN, _TOPK, _SPAN
    from .retention import compute_minimum_reconstructable_bytes
    decoded = decode_compact_sequence(data)
    tokens, spans, header = decoded["tokens"], decoded["spans"], decoded["header"]
    if not tokens:
        raise ValueError("summary-only artifact has no reconstructable token evidence")
    candidate_count = adaptive = rank_overflow = max_k = fallback = 0
    for i, rec in enumerate(tokens):
        if rec.token_index != i:
            raise ValueError("token indexes are not contiguous")
        if rec.effective_topk != len(rec.topk_ids) or len(rec.topk_ids) != len(rec.topk_scores):
            raise ValueError("top-k evidence shape mismatch")
        if rec.effective_topk < header.base_k:
            raise ValueError("floor-starved token evidence")
        candidate_count += rec.effective_topk
        adaptive += rec.effective_topk > header.base_k
        max_k = max(max_k, rec.effective_topk)
        fallback += rec.chosen_rank == 255
        rank_overflow += rec.chosen_rank == 255 or rec.chosen_rank > header.max_adaptive_k
    chunk_count = (len(tokens) + header.chunk_token_capacity - 1) // header.chunk_token_capacity
    meta_len = len(json.dumps(decoded.get("meta", {}), sort_keys=True, separators=(",", ":")).encode())
    # Preserve the legacy floor contract without a second full decode.
    floor = (_HEADER.size + len(header.profile_id.encode()) + meta_len + chunk_count * _CHUNK.size
             + len(tokens) * _TOKEN.size + candidate_count * _TOPK.size + fallback * 4 + len(spans) * _SPAN.size)
    decoded.update(minimum_reconstructable_bytes=floor, stats={
        "token_count": len(tokens), "adaptive_count": adaptive, "max_effective_topk": max_k,
        "rank_overflow": rank_overflow, "span_count": len(spans), "chunk_count": chunk_count,
    }, counters={
        "bytes_read": len(data), "bytes_hashed": None, "token_records_decoded": len(tokens),
        "candidate_entries_decoded": candidate_count, "varint_bytes_decoded": 0,
        "chunks_verified": chunk_count, "span_events_indexed": len(spans),
        "span_overlay_operations": 0, "allocations": None, "full_artifact_materializations": 2,
    })
    return decoded


def verify_evidence_artifact(path: str | Path, *, profile: str = "DSA-CI-Lite", ci_mode: str = "pr", enforce_budget: bool = True, strict_profile_budget: bool = False, enforce_rss_budget: bool = False, audit_keys: dict[int, bytes] | None = None, tension_maps: dict[int, Any] | None = None, verifier_key: bytes | None = None, retention_requirement: str | dict | None = None) -> VerificationReport:
    """Verify local evidence integrity and declared selection; no semantic/DSA-P claim.

    V3.3 uses bounded streaming verification. Legacy versions remain explicitly
    materialized compatibility checks. Completed slow checks produce warnings;
    interrupted resource checks never produce a local success state.
    """
    from .compact_evidence import PREFIX
    from .retention import EvidenceBudgetSummary
    errors, failure_codes, warnings = [], [], []
    prof = get_evidence_profile(profile)
    start = time.perf_counter()
    calibration = None
    owned_trace = not tracemalloc.is_tracing()
    if owned_trace:
        tracemalloc.start()
    decoded = None
    stats: dict[str, int] = {}
    counters: dict[str, Any] = {}
    work_bounds: dict[str, int] = {}
    summary = None
    portable = False
    retention_hash_valid = False
    budget_status = "not_evaluated_disabled" if not enforce_budget else "not_evaluated_due_to_structural_failure"
    try:
        assert_profile_ci_mode(prof, ci_mode)
        artifact_path = find_compact_artifact(path)
        calibration = _calibrate_runtime()
        with artifact_path.open("rb") as source:
            prefix = source.read(PREFIX.size)
        if len(prefix) != PREFIX.size:
            raise ValueError("malformed compact evidence header")
        _, version = PREFIX.unpack(prefix)
        if version == VERSION_V33:
            from .streaming_verifier import verify_stream
            portable = True
            decoded = verify_stream(artifact_path, audit_keys=audit_keys, tension_maps=tension_maps,
                                    counters=counters, work_bounds=work_bounds)
        else:
            # Hard pre-allocation compatibility limit, independent of runner RSS.
            if artifact_path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("legacy materialization exceeds 64 MiB compatibility limit")
            decoded = _legacy_summary(artifact_path.read_bytes(), prof.profile_id.value)
            warnings.append("Legacy compatibility verification materializes evidence; no portable streaming claim.")
        stats, counters = decoded["stats"], decoded["counters"]
        counters = dict(counters)
        _enforce_metadata_binding(_load_run_metadata(path), artifact_path, decoded, decoded["sha256"])
        if decoded["header"].profile_id != prof.profile_id.value:
            raise ValueError(f"artifact profile {decoded['header'].profile_id} does not match requested {prof.profile_id.value}")
        n, size = stats["token_count"], decoded["raw_bytes"]
        if n <= 0:
            raise ValueError("summary-only artifact has no reconstructable token evidence")
        floor = decoded["minimum_reconstructable_bytes"]
        summary = EvidenceBudgetSummary(prof.profile_id.value, n, size, size / n,
                                        prof.ceiling_bytes_per_token, floor, size - floor, size)
        assert_retention_floor(summary)
        if enforce_budget:
            budget_status, budget_errors, budget_codes = _evaluate_profile_budget(summary, prof, strict_profile_budget)
            errors.extend(budget_errors)
            failure_codes.extend(budget_codes)
            if prof.adaptive_record_fraction_limit is not None and stats["adaptive_count"] / n > prof.adaptive_record_fraction_limit:
                raise ValueError("adaptive record fraction exceeds profile limit; escalate evidence profile")
        if retention_requirement is not None:
            # A final requirement is detached and points to the final byte hash.
            req = json.loads(retention_requirement) if isinstance(retention_requirement, str) else retention_requirement
            if not isinstance(req, dict) or req.get("artifact_content_hash") != decoded["sha256"]:
                raise ValueError("detached retention requirement content hash mismatch")
            if req.get("chunk_merkle_root") != getattr(decoded.get("manifest"), "chunk_merkle_root", None):
                raise ValueError("detached retention requirement Merkle root mismatch")
            if req.get("minimum_reconstructable_bytes") != floor:
                raise ValueError("detached retention requirement floor mismatch")
            retention_hash_valid = True
            warnings.append("Retention content binding checked; signature and independent storage receipt are not verified by this local command.")
    except (MemoryError, TimeoutError) as exc:
        errors.append(f"verification incomplete due to resource instability: {type(exc).__name__}")
        failure_codes.append(AST_INFRASTRUCTURE_INSTABILITY)
    except Exception as exc:
        errors.append(str(exc))
        msg = str(exc).lower()
        if any(word in msg for word in ("audit key", "keyed", "commitment", "replay")):
            failure_codes.append(525)
        elif "work bound" in msg or "streaming limit" in msg:
            failure_codes.append(AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION)
        else:
            failure_codes.append(AST_RETENTION_FLOOR_VIOLATION if "floor" in msg or "summary-only" in msg else AST_SCHEMA_MISMATCH)
    finally:
        if counters:
            counters["bytes_read"] += PREFIX.size  # version dispatch read, including failed verification
        _, peak = tracemalloc.get_traced_memory()
        if owned_trace:
            tracemalloc.stop()
    elapsed = time.perf_counter() - start
    rss = _rss_bytes()
    rss_limit = int(prof.verifier_peak_rss_mib or prof.verifier_peak_mib) * 1024 * 1024
    if elapsed > prof.verifier_time_seconds:
        warnings.append(f"verification elapsed {elapsed:.6f}s exceeds advisory SLO {prof.verifier_time_seconds:.6f}s")
    if enforce_rss_budget and rss > rss_limit:
        warnings.append(f"observed RSS {rss} exceeds advisory profile RSS {rss_limit}")
    traced_limit = int(prof.verifier_traced_peak_mib or prof.verifier_peak_mib) * 1024 * 1024
    if peak > traced_limit:
        warnings.append(f"observed traced allocation {peak} exceeds advisory profile allocation {traced_limit}")
    cert = None
    if counters:
        names = ("bytes_read", "bytes_hashed", "token_records_decoded", "candidate_entries_decoded",
                 "varint_bytes_decoded", "chunks_verified", "span_events_indexed", "span_overlay_operations",
                 "allocations", "full_artifact_materializations")
        cert = VerifierWorkCertificate(**{name: counters.get(name) for name in names},
            maximum_live_bytes=counters.get("maximum_live_bytes", peak),
            certificate_kind="v33_streaming_work" if portable else "legacy_observations",
            work_completed=decoded is not None,
            artifact_sha256=decoded['sha256'] if decoded is not None else None,
            work_bounds=work_bounds)
        if verifier_key is not None:
            if not verifier_key:
                errors.append("work certificate signing key must not be empty")
                failure_codes.append(AST_SCHEMA_MISMATCH)
            else:
                cert = replace(cert, signature=sign_work_certificate(cert, verifier_key))
    ok = not errors
    outcome = "pass" if ok else "fail"
    if errors and failure_codes and all(code == AST_INFRASTRUCTURE_INSTABILITY for code in failure_codes):
        outcome = "INCONCLUSIVE_INFRA"
    n = stats.get("token_count", 0)
    return VerificationReport(
        ok=ok, profile_id=prof.profile_id.value, token_count=n, elapsed_seconds=elapsed,
        peak_tracemalloc_bytes=peak, verifier_peak_rss_bytes=rss, verifier_peak_rss_limit_bytes=rss_limit,
        raw_bytes_per_token=summary.raw_bytes_per_token if summary else 0.0, compressed_bytes_per_token=None,
        adaptive_record_count=stats.get("adaptive_count", 0),
        adaptive_record_fraction=stats.get("adaptive_count", 0) / n if n else 0.0,
        max_effective_topk=stats.get("max_effective_topk", 0), rank_overflow_count=stats.get("rank_overflow", 0),
        retained_reconstructable_bytes=summary.retained_reconstructable_bytes if summary else 0,
        minimum_reconstructable_bytes=summary.minimum_reconstructable_bytes if summary else 0,
        retention_floor_margin_bytes=summary.retention_floor_margin if summary else 0,
        verifier_reconstruction_seconds_mean=0.0, verifier_reconstruction_seconds_p50=0.0,
        verifier_reconstruction_seconds_p95=0.0, tokens_reconstructed_per_second=n / elapsed if elapsed else 0.0,
        chunks_reconstructed=stats.get("chunk_count", 0), span_events_overlaid=0,
        adaptive_records_reconstructed=stats.get("adaptive_count", 0), budget_status=budget_status,
        verification_outcome=outcome, failure_codes=sorted(set(failure_codes), key=str), errors=errors,
        strict_profile_budget=strict_profile_budget, minimum_budget_token_count=prof.minimum_budget_token_count,
        ceiling_bytes_per_token=prof.ceiling_bytes_per_token, work_certificate=cert,
        retention_state="LOCAL_PASS" if ok else "NOT_VERIFIED", retention_hash_valid=retention_hash_valid,
        warnings=warnings, calibration_seconds=calibration,
        normalized_runtime_ratio=elapsed / calibration if calibration else None,
        runtime_calibration_id="fixed-candidate-sha256-v1" if calibration else None,
    )
