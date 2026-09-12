from dataclasses import replace

from dualstream.work_certificate import (
    CERTIFICATE_VERSION,
    VerifierRuntimeDiagnostics,
    VerifierWorkCertificate,
    WorkEnvelope,
    canonical_certificate_payload,
    certificate_digest,
    sign_work_certificate,
    validate_work_envelope,
    verify_work_certificate_signature,
)


def certificate(**overrides):
    values = dict(
        certificate_version=CERTIFICATE_VERSION,
        work_profile_id="portable-work-v1",
        artifact_sha256="ab" * 32,
        bytes_read=1000,
        bytes_hashed=2000,
        token_records_decoded=10,
        candidate_entries_decoded=30,
        varint_bytes_decoded=0,
        chunks_verified=1,
        span_events_indexed=2,
        span_overlay_operations=4,
        allocations=20,
        maximum_live_bytes=4096,
        full_artifact_materializations=0,
    )
    values.update(overrides)
    return VerifierWorkCertificate(**values)


def test_canonical_certificate_is_stable_and_signature_is_not_self_referential():
    cert = certificate()
    signed = sign_work_certificate(cert, b"verifier-key")
    assert signed.signature is not None
    assert canonical_certificate_payload(cert) == canonical_certificate_payload(signed)
    assert certificate_digest(cert) == certificate_digest(signed)
    assert verify_work_certificate_signature(signed, b"verifier-key")
    assert not verify_work_certificate_signature(signed, b"wrong-key")


def test_runtime_diagnostics_cannot_change_certificate_identity():
    cert = certificate()
    fast = VerifierRuntimeDiagnostics(0.01, 1000, 2000)
    slow = VerifierRuntimeDiagnostics(99.0, 900000, 800000)
    assert fast != slow
    assert certificate_digest(cert) == certificate_digest(cert)


def test_tampering_changes_digest_and_breaks_signature():
    signed = sign_work_certificate(certificate(), b"key")
    tampered = replace(signed, candidate_entries_decoded=31)
    assert certificate_digest(signed) != certificate_digest(tampered)
    assert not verify_work_certificate_signature(tampered, b"key")


def test_work_envelope_accepts_deterministic_counts_within_bounds():
    cert = certificate()
    envelope = WorkEnvelope(work_profile_id="portable-work-v1")
    assert validate_work_envelope(
        cert, envelope, artifact_size=1000, token_count=10, span_count=2
    ) == []


def test_work_envelope_rejects_prohibited_materialization_and_excess_candidates():
    cert = certificate(full_artifact_materializations=1, candidate_entries_decoded=400)
    envelope = WorkEnvelope(work_profile_id="portable-work-v1")
    violations = validate_work_envelope(
        cert, envelope, artifact_size=1000, token_count=10, span_count=2
    )
    assert any("full_artifact_materializations" in v for v in violations)
    assert any("candidate_entries_decoded" in v for v in violations)


def test_work_profile_mismatch_fails_closed():
    violations = validate_work_envelope(
        certificate(work_profile_id="other"),
        WorkEnvelope(work_profile_id="portable-work-v1"),
        artifact_size=1000,
        token_count=10,
        span_count=2,
    )
    assert violations == [
        "certificate work profile other does not match envelope portable-work-v1"
    ]
