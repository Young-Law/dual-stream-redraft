from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, replace, field


CERTIFICATE_VERSION = 2


@dataclass(frozen=True)
class VerifierWorkCertificate:
    """Portable, deterministic description of verifier work.

    Every field in this object MUST be derivable from the artifact, verifier
    algorithm, and declared work profile. Host measurements such as elapsed
    time, RSS, and tracemalloc peaks intentionally do not belong here: putting
    them in the signed payload would make the same artifact produce different
    certificates on different machines.
    """

    certificate_version: int
    work_profile_id: str
    artifact_sha256: str | None
    bytes_read: int
    bytes_hashed: int | None
    token_records_decoded: int
    candidate_entries_decoded: int
    varint_bytes_decoded: int
    chunks_verified: int
    span_events_indexed: int
    span_overlay_operations: int
    allocations: int | None
    maximum_live_bytes: int | None
    full_artifact_materializations: int
    signature: str | None = None
    work_completed: bool = True
    work_bounds: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierRuntimeDiagnostics:
    """Environment-sensitive diagnostics; never part of certificate signing."""

    elapsed_seconds: float
    peak_tracemalloc_bytes: int
    peak_rss_bytes: int


@dataclass(frozen=True)
class WorkEnvelope:
    """Hard deterministic limits for one portable verifier work profile."""

    work_profile_id: str
    max_bytes_read_per_artifact_byte: int = 4
    max_bytes_hashed_per_artifact_byte: int = 8
    max_candidate_entries_per_token: int = 32
    max_span_overlay_operations_per_span: int = 64
    max_allocations_per_span: int = 0
    fixed_allocations: int = 0
    max_allocations_per_token: int = 8
    max_live_bytes_per_token: int = 512
    fixed_live_bytes: int = 1 << 20
    max_full_artifact_materializations: int = 0


def canonical_certificate_payload(cert: VerifierWorkCertificate) -> bytes:
    """Return canonical bytes for equality, hashing, and signatures.

    `signature` is excluded by construction. The function rejects unsupported
    versions rather than silently changing canonicalization semantics.
    """

    if cert.certificate_version not in (1, CERTIFICATE_VERSION):
        raise ValueError(
            f"unsupported work certificate version: {cert.certificate_version}"
        )
    data = asdict(cert)
    data.pop("signature", None)
    if cert.certificate_version == 1:
        if not cert.work_completed or cert.work_bounds:
            raise ValueError("version 1 cannot represent partial work or parser bounds")
        data.pop("work_completed")
        data.pop("work_bounds")
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def certificate_digest(cert: VerifierWorkCertificate) -> str:
    return hashlib.sha256(canonical_certificate_payload(cert)).hexdigest()


def sign_work_certificate(cert: VerifierWorkCertificate, key: bytes) -> VerifierWorkCertificate:
    if not key:
        raise ValueError("work certificate signing key must not be empty")
    signature = hmac.new(key, canonical_certificate_payload(cert), hashlib.sha256).hexdigest()
    return replace(cert, signature=signature)


def verify_work_certificate_signature(cert: VerifierWorkCertificate, key: bytes) -> bool:
    if cert.signature is None:
        return False
    expected = hmac.new(key, canonical_certificate_payload(cert), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, cert.signature)


def validate_work_envelope(
    cert: VerifierWorkCertificate,
    envelope: WorkEnvelope,
    *,
    artifact_size: int,
    token_count: int,
    span_count: int,
) -> list[str]:
    """Return deterministic work-envelope violations.

    Empty output means the certificate is within the declared portable work
    envelope. Runtime/RSS are deliberately absent from this function.
    """

    if cert.work_profile_id != envelope.work_profile_id:
        return [
            f"certificate work profile {cert.work_profile_id} does not match "
            f"envelope {envelope.work_profile_id}"
        ]

    limits = (
        ("bytes_read", cert.bytes_read, artifact_size * envelope.max_bytes_read_per_artifact_byte),
        ("bytes_hashed", cert.bytes_hashed, artifact_size * envelope.max_bytes_hashed_per_artifact_byte),
        (
            "candidate_entries_decoded",
            cert.candidate_entries_decoded,
            token_count * envelope.max_candidate_entries_per_token,
        ),
        (
            "span_overlay_operations",
            cert.span_overlay_operations,
            span_count * envelope.max_span_overlay_operations_per_span,
        ),
        ("allocations", cert.allocations, envelope.fixed_allocations + token_count * envelope.max_allocations_per_token + span_count * envelope.max_allocations_per_span),
        (
            "maximum_live_bytes",
            cert.maximum_live_bytes,
            envelope.fixed_live_bytes + token_count * envelope.max_live_bytes_per_token,
        ),
        (
            "full_artifact_materializations",
            cert.full_artifact_materializations,
            envelope.max_full_artifact_materializations,
        ),
    )
    return [f"{name} {observed} exceeds deterministic limit {limit}" for name, observed, limit in limits if observed is None or observed < 0 or observed > limit]


def streaming_work_envelope(profile) -> WorkEnvelope:
    """Pinned v2 limits: one integrity pass plus at most four replay passes.

    Allocation events count explicit wire reads, not Python allocator calls.
    Logical live bytes bound retained wire buffers, not decoded object storage.
    Chunk/metadata caps separately bound decoded residency.
    """
    return WorkEnvelope(
        work_profile_id=profile.verifier_work_profile_id + ":streaming-v2",
        max_bytes_read_per_artifact_byte=6,
        max_bytes_hashed_per_artifact_byte=12,
        max_candidate_entries_per_token=profile.max_adaptive_k * 5,
        max_allocations_per_token=16,
        fixed_allocations=64,
        max_allocations_per_span=3,
    )
