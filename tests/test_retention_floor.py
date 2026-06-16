import pytest
from dualstream.compact_evidence import encode_compact_sequence
from dualstream.retention import compute_evidence_budget_summary, assert_evidence_budget, assert_retention_floor


def fixture(n=100):
    return encode_compact_sequence([{"chosen_id": i, "topk_ids": [i, i+1, i+2], "topk_scores": [.7,.2,.1]} for i in range(n)])


def test_lite_ceiling_and_floor_pass_for_compact_fixture():
    summary = compute_evidence_budget_summary(fixture(1000), "DSA-CI-Lite")
    assert_evidence_budget(summary)
    assert_retention_floor(summary)


def test_summary_only_or_truncated_fails():
    with pytest.raises(Exception):
        compute_evidence_budget_summary(b'{"summary":true}', "DSA-CI-Lite")
    with pytest.raises(Exception):
        compute_evidence_budget_summary(fixture(5)[:-5], "DSA-CI-Lite")


def test_lite_ceiling_passes_for_ten_thousand_token_fixture():
    summary = compute_evidence_budget_summary(fixture(10000), "DSA-CI-Lite")
    assert summary.raw_bytes_per_token <= 24
    assert_evidence_budget(summary)
    assert_retention_floor(summary)
