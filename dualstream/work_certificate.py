from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, replace


CERTIFICATE_VERSION = 1


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
    artifact_sha256: str
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
    signature: str | None = None


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
    max_allocations_per_token: int = 8
    max_live_bytes_per_token: int = 512
    fixed_live_bytes: int = 1 << 20
    max_full_artifact_materializations: int = 0


def canonical_certificate_payload(cert: VerifierWorkCertificate) -> bytes:
    """Return canonical bytes for equality, hashing, and signatures.

    `signature` is excluded by construction. The function rejects unsupported
    versions rather than silently changing canonicalization semantics.
    """

    if cert.certificate_version != CERTIFICATE_VERSION:
        raise ValueError(
            f"unsupported work certificate version: {cert.certificate_version}"
        )
    data = asdict(cert)
    data.pop("signature", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def certificate_digest(cert: VerifierWorkCertificate) -> str:
    return hashlib.sha256(canonical_certificate_payload(cert)).hexdigest()


def sign_work_certificate(cert: VerifierWorkCertificate, key: bytes) -> VerifierWorkCertificate:
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
        ("allocations", cert.allocations, token_count * envelope.max_allocations_per_token),
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
    return [f"{name} {observed} exceeds deterministic limit {limit}" for name, observed, limit in limits if observed > limit]
