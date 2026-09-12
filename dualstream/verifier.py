from __future__ import annotations

import json
import os
import time
import tracemalloc
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .compact_evidence import MAGIC, PREFIX, decode_compact_sequence, verify_keyed_replay, compute_retention_requirement_hash, EvidenceManifestV33, VERSION_V33
from .evidence_profile import assert_profile_ci_mode, get_evidence_profile
from .retention import compute_evidence_budget_summary, assert_retention_floor
from .streaming_verifier import verify_v33_stream
from .work_certificate import (
    CERTIFICATE_VERSION,
    VerifierRuntimeDiagnostics,
    VerifierWorkCertificate,
    WorkEnvelope,
    canonical_certificate_payload as canonical_serialize_certificate,
    sign_work_certificate,
    validate_work_envelope,
    verify_work_certificate_signature as _verify_work_certificate_signature,
)
from .vocab import (
    AST_RETENTION_FLOOR_VIOLATION,
    AST_VERIFIER_RESOURCE_BUDGET_EXCEEDED,
    AST_SCHEMA_MISMATCH,
    AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION,
    AST_INFRASTRUCTURE_INSTABILITY,
)


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
    runtime_diagnostics: VerifierRuntimeDiagnostics | None = None
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


def verify_work_certificate_signature(
    cert: VerifierWorkCertificate,
    signature_or_key: str | bytes,
    key: bytes | None = None,
) -> bool:
    """Verify canonical certificates, accepting the pre-v2.10 call shape too."""
    if key is None:
        return _verify_work_certificate_signature(cert, signature_or_key)  # type: ignore[arg-type]
    if not isinstance(signature_or_key, str) or signature_or_key != cert.signature:
        return False
    return _verify_work_certificate_signature(cert, key)


def verify_evidence_artifact(path: str | Path, *, profile: str = "DSA-CI-Lite", ci_mode: str = "pr", enforce_budget: bool = True, strict_profile_budget: bool = False, enforce_rss_budget: bool = False, audit_keys: dict[int, bytes] | None = None, tension_maps: dict[int, Any] | None = None, verifier_key: bytes | None = None, retention_requirement: str | dict | None = None) -> VerificationReport:
    errors: list[str] = []
    failure_codes: list[int | str] = []
    start = time.perf_counter()
    tracemalloc.start()
    prof = get_evidence_profile(profile)
    token_count = adaptive_count = max_eff = rank_overflow = chunks = spans = 0
    raw_bpt = 0.0
    compressed_bpt = None
    retained = minimum = margin = 0
    budget_status = "not_evaluated_disabled" if not enforce_budget else "not_evaluated_due_to_structural_failure"
    cert: VerifierWorkCertificate | None = None
    manifest: EvidenceManifestV33 | None = None
    artifact_sha256 = "0" * 64
    artifact_size = 0

    try:
        assert_profile_ci_mode(prof, ci_mode)
        artifact_path = find_compact_artifact(path)
        artifact_size = artifact_path.stat().st_size
        with artifact_path.open("rb") as prefix_stream:
            prefix = prefix_stream.read(PREFIX.size)
        is_v33 = len(prefix) == PREFIX.size and PREFIX.unpack(prefix) == (MAGIC, VERSION_V33)

        if is_v33:
            streamed = verify_v33_stream(
                artifact_path, audit_keys=audit_keys, initial_bytes_read=len(prefix)
            )
            header = streamed.header
            manifest = streamed.manifest
            artifact_sha256 = streamed.artifact_sha256
            token_count = header.token_count
            chunks = manifest.chunk_count
            spans = manifest.span_event_count
            adaptive_count = streamed.adaptive_record_count
            max_eff = streamed.max_effective_topk
            rank_overflow = streamed.rank_overflow_count
            raw_bpt = artifact_size / token_count if token_count else 0.0
            retained = manifest.raw_evidence_bytes
            minimum = manifest.minimum_reconstructable_bytes
            margin = retained - minimum
            if token_count <= 0:
                raise ValueError("summary-only artifact has no reconstructable token evidence")
            if retained < minimum:
                raise ValueError("retained compact evidence is below the reconstructable floor")
            decoded_binding = {"header": header}
            _enforce_metadata_binding(
                _load_run_metadata(path), artifact_path, decoded_binding, artifact_sha256
            )
            if header.profile_id != prof.profile_id.value:
                raise ValueError(
                    f"artifact profile {header.profile_id} does not match requested {prof.profile_id.value}"
                )
            work = streamed.counters
            cert = VerifierWorkCertificate(
                certificate_version=CERTIFICATE_VERSION,
                work_profile_id=prof.verifier_work_profile_id,
                artifact_sha256=artifact_sha256,
                bytes_read=work.bytes_read,
                bytes_hashed=work.bytes_hashed,
                token_records_decoded=work.token_records_decoded,
                candidate_entries_decoded=work.candidate_entries_decoded,
                varint_bytes_decoded=work.varint_bytes_decoded,
                chunks_verified=work.chunks_verified,
                span_events_indexed=work.span_events_indexed,
                span_overlay_operations=work.span_overlay_operations,
                allocations=work.allocations,
                maximum_live_bytes=work.maximum_live_bytes,
                full_artifact_materializations=work.full_artifact_materializations,
            )
            if enforce_budget:
                envelope = WorkEnvelope(
                    work_profile_id=prof.verifier_work_profile_id,
                    max_candidate_entries_per_token=prof.max_adaptive_k * (2 if audit_keys is not None else 1),
                )
                violations = validate_work_envelope(
                    cert, envelope, artifact_size=artifact_size,
                    token_count=token_count, span_count=spans,
                )
                if violations:
                    errors.extend(violations)
                    failure_codes.append(AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION)
        else:
            # V3.1/V3.2 remain readable for compatibility. Their legacy codec
            # materializes the artifact and therefore does not claim portable
            # zero-materialization conformance.
            data = artifact_path.read_bytes()
            artifact_sha256 = __import__("hashlib").sha256(data).hexdigest()
            decoded = decode_compact_sequence(data)
            if audit_keys is not None:
                verify_keyed_replay(decoded, audit_keys)
            _enforce_metadata_binding(_load_run_metadata(path), artifact_path, decoded, artifact_sha256)
            if decoded["header"].profile_id != prof.profile_id.value:
                raise ValueError(
                    f"artifact profile {decoded['header'].profile_id} does not match requested {prof.profile_id.value}"
                )
            records = decoded["tokens"]
            token_count = len(records)
            spans = len(decoded.get("spans", []))
            chunks = (token_count + decoded["header"].chunk_token_capacity - 1) // decoded["header"].chunk_token_capacity
            effective = [record.effective_topk for record in records]
            max_eff = max(effective, default=0)
            adaptive_count = sum(value > prof.base_k for value in effective)
            rank_overflow = sum(
                record.chosen_rank == 255 or record.chosen_rank > prof.max_adaptive_k
                for record in records
            )
            summary = compute_evidence_budget_summary(data, prof.profile_id.value)
            assert_retention_floor(summary)
            raw_bpt = summary.raw_bytes_per_token
            compressed_bpt = round(len(zlib.compress(data)) / token_count, 2) if token_count else None
            retained = summary.retained_reconstructable_bytes
            minimum = summary.minimum_reconstructable_bytes
            margin = summary.retention_floor_margin
            cert = VerifierWorkCertificate(
                certificate_version=CERTIFICATE_VERSION,
                work_profile_id="legacy-materializing-v1",
                artifact_sha256=artifact_sha256,
                bytes_read=len(data), bytes_hashed=len(data),
                token_records_decoded=token_count,
                candidate_entries_decoded=sum(effective),
                varint_bytes_decoded=0, chunks_verified=chunks,
                span_events_indexed=spans, span_overlay_operations=0,
                allocations=token_count, maximum_live_bytes=len(data),
                full_artifact_materializations=1,
            )

        if enforce_budget and token_count:
            if not strict_profile_budget and token_count < prof.minimum_budget_token_count:
                budget_status = "not_evaluated_short_fixture"
            elif raw_bpt > prof.ceiling_bytes_per_token:
                budget_status = "fail"
                errors.append(
                    f"raw bytes/token {raw_bpt:.3f} exceeds ceiling {prof.ceiling_bytes_per_token}"
                )
                failure_codes.append(PROFILE_BYTE_BUDGET_EXCEEDED)
            else:
                budget_status = "pass"
            if (
                prof.adaptive_record_fraction_limit is not None
                and adaptive_count / token_count > prof.adaptive_record_fraction_limit
            ):
                raise ValueError("adaptive record fraction exceeds profile limit")
    except Exception as exc:
        errors.append(str(exc))
        message = str(exc).lower()
        failure_codes.append(
            AST_RETENTION_FLOOR_VIOLATION
            if "floor" in message or "summary-only" in message
            else AST_SCHEMA_MISMATCH
        )

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start
    rss = _rss_bytes()
    rss_limit = int(prof.verifier_peak_rss_mib or prof.verifier_peak_mib) * 1024 * 1024
    runtime_diagnostics = VerifierRuntimeDiagnostics(elapsed, peak, rss)

    if cert is not None and verifier_key is not None:
        cert = sign_work_certificate(cert, verifier_key)

    if enforce_budget:
        traced_limit = int(prof.verifier_traced_peak_mib or prof.verifier_peak_mib) * 1024 * 1024
        if peak > traced_limit:
            errors.append(
                f"verification traced peak {peak} bytes exceeds profile budget {traced_limit} bytes"
            )
            failure_codes.append(AST_INFRASTRUCTURE_INSTABILITY)
        if elapsed > prof.verifier_time_seconds:
            errors.append(
                f"verification elapsed {elapsed:.6f}s exceeds profile budget {prof.verifier_time_seconds:.6f}s"
            )
            failure_codes.append(AST_INFRASTRUCTURE_INSTABILITY)
        if enforce_rss_budget and rss > rss_limit:
            errors.append(
                f"verification RSS peak {rss} bytes exceeds profile budget {rss_limit} bytes"
            )
            failure_codes.append(AST_INFRASTRUCTURE_INSTABILITY)

    retention_hash_valid = False
    if manifest is not None and retention_requirement is not None:
        expected_hash = compute_retention_requirement_hash(retention_requirement)
        retention_hash_valid = expected_hash.hex() == manifest.retention_requirement_hash
        if not retention_hash_valid:
            errors.append(
                f"retention requirement hash mismatch: expected {expected_hash.hex()}, "
                f"manifest has {manifest.retention_requirement_hash}"
            )
            failure_codes.append(AST_SCHEMA_MISMATCH)

    ok = not errors
    outcome = "pass" if ok else "fail"
    if not ok and failure_codes and all(
        code == AST_INFRASTRUCTURE_INSTABILITY for code in failure_codes
    ):
        outcome = "INCONCLUSIVE_INFRA"
    tps = token_count / elapsed if elapsed > 0 else 0.0
    return VerificationReport(
        ok=ok, profile_id=prof.profile_id.value, token_count=token_count,
        elapsed_seconds=elapsed, peak_tracemalloc_bytes=peak,
        verifier_peak_rss_bytes=rss, verifier_peak_rss_limit_bytes=rss_limit,
        raw_bytes_per_token=raw_bpt, compressed_bytes_per_token=compressed_bpt,
        adaptive_record_count=adaptive_count,
        adaptive_record_fraction=adaptive_count / token_count if token_count else 0.0,
        max_effective_topk=max_eff, rank_overflow_count=rank_overflow,
        retained_reconstructable_bytes=retained,
        minimum_reconstructable_bytes=minimum,
        retention_floor_margin_bytes=margin,
        verifier_reconstruction_seconds_mean=0.0,
        verifier_reconstruction_seconds_p50=0.0,
        verifier_reconstruction_seconds_p95=0.0,
        tokens_reconstructed_per_second=tps, chunks_reconstructed=chunks,
        span_events_overlaid=spans, adaptive_records_reconstructed=adaptive_count,
        budget_status=budget_status, verification_outcome=outcome,
        failure_codes=sorted(set(failure_codes), key=str), errors=errors,
        strict_profile_budget=strict_profile_budget,
        minimum_budget_token_count=prof.minimum_budget_token_count,
        ceiling_bytes_per_token=prof.ceiling_bytes_per_token,
        work_certificate=cert, runtime_diagnostics=runtime_diagnostics,
        retention_state="LOCAL_PASS" if ok else "LOCAL_FAIL",
        retention_hash_valid=retention_hash_valid,
    )
