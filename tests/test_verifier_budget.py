from dualstream.compact_evidence import encode_compact_sequence
from dualstream.verifier import verify_evidence_artifact


def test_verifier_metrics_present(tmp_path):
    blob = encode_compact_sequence([{"chosen_id": i, "topk_ids": [i, i+1, i+2], "topk_scores": [.7,.2,.1]} for i in range(500)])
    p = tmp_path / "compact_evidence.dsae"
    p.write_bytes(blob)
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert report.ok, report.errors
    assert report.elapsed_seconds >= 0
    assert report.peak_tracemalloc_bytes >= 0
    assert report.peak_rss_bytes >= 0
    assert report.raw_bytes_per_token > 0
    assert report.adaptive_record_fraction == 0
    assert report.max_effective_topk == 3
    assert report.retention_floor_margin >= 0


def test_forensic_rejected_for_pr(tmp_path):
    p = tmp_path / "compact_evidence.dsae"
    p.write_bytes(encode_compact_sequence([{"chosen_id": 1, "topk_ids": [1,2,3], "topk_scores": [.7,.2,.1]}], profile="DSA-Forensic"))
    report = verify_evidence_artifact(tmp_path, profile="DSA-Forensic", ci_mode="pr")
    assert not report.ok


def test_verifier_compute_budget_failure_is_enforced(tmp_path, monkeypatch):
    from dataclasses import replace
    import dualstream.verifier as verifier
    from dualstream.evidence_profile import get_evidence_profile

    p = tmp_path / "compact_evidence.dsae"
    p.write_bytes(encode_compact_sequence([{"chosen_id": i, "topk_ids": [i, i+1, i+2], "topk_scores": [.7,.2,.1]} for i in range(20)]))
    tiny = replace(get_evidence_profile("DSA-CI-Lite"), verifier_time_seconds=0.0, verifier_peak_mib=1)
    monkeypatch.setattr(verifier, "get_evidence_profile", lambda _profile: tiny)
    monkeypatch.setattr(verifier, "assert_profile_ci_mode", lambda _profile, _ci_mode: tiny)
    report = verifier.verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert not report.ok
    assert any("exceeds profile budget" in err for err in report.errors)


def _over_budget_rows(n):
    return [{"chosen_id": i + 100000, "topk_ids": [i, i+1, i+2], "topk_scores": [.7,.2,.1]} for i in range(n)]


def test_strict_short_fixture_over_budget_reports_budget_fail(tmp_path):
    (tmp_path / "compact_evidence.dsae").write_bytes(encode_compact_sequence(_over_budget_rows(4), adaptive_k=False))
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr", strict_profile_budget=True)
    assert not report.ok
    assert report.budget_status == "fail"
    assert "profile_byte_budget_exceeded" in report.failure_codes


def test_non_strict_short_fixture_reports_not_evaluated_short_fixture(tmp_path):
    (tmp_path / "compact_evidence.dsae").write_bytes(encode_compact_sequence(_over_budget_rows(4), adaptive_k=False))
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert report.ok, report.errors
    assert report.budget_status == "not_evaluated_short_fixture"


def test_canonical_over_budget_reports_budget_fail(tmp_path):
    (tmp_path / "compact_evidence.dsae").write_bytes(encode_compact_sequence(_over_budget_rows(10000), adaptive_k=False))
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert not report.ok
    assert report.budget_status == "fail"
    assert "profile_byte_budget_exceeded" in report.failure_codes


def test_structural_failure_budget_status(tmp_path):
    (tmp_path / "compact_evidence.dsae").write_bytes(b"not compact")
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert not report.ok
    assert report.budget_status == "not_evaluated_due_to_structural_failure"

def test_infrastructure_instability_yields_inconclusive_infra(tmp_path, monkeypatch):
    from dataclasses import replace
    import dualstream.verifier as verifier
    from dualstream.evidence_profile import get_evidence_profile

    p = tmp_path / "compact_evidence.dsae"
    p.write_bytes(encode_compact_sequence([{"chosen_id": i, "topk_ids": [i, i+1, i+2], "topk_scores": [.7,.2,.1]} for i in range(20)]))
    
    # Tiny time budget, but large traced memory budget so it only fails on elapsed time.
    tiny = replace(get_evidence_profile("DSA-CI-Lite"), verifier_time_seconds=0.0, verifier_traced_peak_mib=1024, verifier_peak_mib=1024)
    monkeypatch.setattr(verifier, "get_evidence_profile", lambda _profile: tiny)
    monkeypatch.setattr(verifier, "assert_profile_ci_mode", lambda _profile, _ci_mode: tiny)
    
    report = verifier.verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert not report.ok
    assert report.verification_outcome == "INCONCLUSIVE_INFRA"
    assert 524 in report.failure_codes # AST_INFRASTRUCTURE_INSTABILITY
    assert 523 not in report.failure_codes # AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION


def test_verifier_work_certificate_signing_and_verification(tmp_path):
    from dualstream.verifier import sign_work_certificate, verify_work_certificate_signature
    from dataclasses import replace

    blob = encode_compact_sequence([{"chosen_id": i, "topk_ids": [i, i+1, i+2], "topk_scores": [.7,.2,.1]} for i in range(100)])
    p = tmp_path / "compact_evidence.dsae"
    p.write_bytes(blob)
    
    key = b"secret-verifier-pki-hmac-key"
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr", verifier_key=key)
    assert report.ok, report.errors
    cert = report.work_certificate
    assert cert is not None
    assert cert.signature is not None
    
    # 1. Validate the signature directly using our helper
    assert verify_work_certificate_signature(cert, cert.signature, key) is True
    
    # 2. Check that using a wrong key fails validation
    assert verify_work_certificate_signature(cert, cert.signature, b"wrong-key") is False
    
    # 3. Check that tampering with any field invalidates the signature
    tampered_cert = replace(cert, bytes_read=cert.bytes_read + 1)
    assert verify_work_certificate_signature(tampered_cert, cert.signature, key) is False
