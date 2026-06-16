from dualstream.compact_evidence import encode_compact_sequence, reconstruct_token_evidence
from dualstream.verifier import verify_evidence_artifact


def test_v29_pipeline_compact_passes(tmp_path):
    rows = [{"chosen_id": i+10, "topk_ids": [i+10, i+20, i+30], "topk_scores": [.6,.3,.1]} for i in range(1000)]
    blob = encode_compact_sequence(rows, profile="DSA-CI-Lite", adaptive_k=True, max_adaptive_k=5)
    assert len(reconstruct_token_evidence(blob)) == len(rows)
    (tmp_path / "compact_evidence.dsae").write_bytes(blob)
    report = verify_evidence_artifact(tmp_path, profile="DSA-CI-Lite", ci_mode="pr")
    assert report.ok, report.errors
