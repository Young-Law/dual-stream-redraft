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
