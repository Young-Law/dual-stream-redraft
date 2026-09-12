import hashlib
from pathlib import Path

import dualstream.verifier as verifier
from dualstream.compact_evidence import VERSION_V33, encode_compact_sequence
from dualstream.vocab import AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION
from dualstream.work_certificate import canonical_certificate_payload


def _artifact(token_count: int = 512) -> bytes:
    return encode_compact_sequence(
        [
            {
                "chosen_id": index,
                "topk_ids": [index, index + 1, index + 2],
                "topk_scores": [0.7, 0.2, 0.1],
            }
            for index in range(token_count)
        ],
        wire_version=VERSION_V33,
        adaptive_k=False,
    )


def test_v33_verifier_does_not_use_full_file_materialization(tmp_path, monkeypatch):
    path = tmp_path / "artifact.dsae"
    path.write_bytes(_artifact())

    def prohibited_read_bytes(_path):
        raise AssertionError("V3.3 verifier must not call Path.read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", prohibited_read_bytes)
    report = verifier.verify_evidence_artifact(path)

    assert report.ok, report.errors
    assert report.work_certificate.full_artifact_materializations == 0
    assert report.work_certificate.bytes_read == path.stat().st_size + 10
    assert report.work_certificate.artifact_sha256 == hashlib.sha256(_artifact()).hexdigest()


def test_integrated_certificate_is_identical_across_runs(tmp_path):
    path = tmp_path / "artifact.dsae"
    path.write_bytes(_artifact())

    first = verifier.verify_evidence_artifact(path)
    second = verifier.verify_evidence_artifact(path)

    assert first.ok and second.ok
    assert canonical_certificate_payload(first.work_certificate) == canonical_certificate_payload(second.work_certificate)
    assert first.runtime_diagnostics is not None
    assert second.runtime_diagnostics is not None


def test_streaming_live_buffer_is_smaller_than_artifact(tmp_path):
    path = tmp_path / "artifact.dsae"
    path.write_bytes(_artifact(10_000))
    report = verifier.verify_evidence_artifact(path)

    assert report.ok, report.errors
    assert report.work_certificate.maximum_live_bytes < path.stat().st_size


def test_deterministic_envelope_violation_maps_to_ast_523(tmp_path, monkeypatch):
    path = tmp_path / "artifact.dsae"
    path.write_bytes(_artifact())
    monkeypatch.setattr(
        verifier,
        "validate_work_envelope",
        lambda *args, **kwargs: ["deterministic test violation"],
    )

    report = verifier.verify_evidence_artifact(path)

    assert not report.ok
    assert report.verification_outcome == "fail"
    assert AST_DETERMINISTIC_VERIFIER_WORK_VIOLATION in report.failure_codes
    assert "deterministic test violation" in report.errors


def test_keyed_replay_uses_bounded_second_pass(tmp_path):
    key = b"streaming-replay-key"
    rows = [
        {
            "chosen_id": index,
            "topk_ids": list(range(index, index + 10)),
            "topk_scores": [0.2, 0.18, 0.16, 0.14, 0.1, 0.08, 0.05, 0.04, 0.03, 0.02],
        }
        for index in range(64)
    ]
    blob = encode_compact_sequence(
        rows, wire_version=VERSION_V33, adaptive_k=True,
        audit_key=key, audit_key_id=7, stochastic_rate_ppm=1,
    )
    path = tmp_path / "artifact.dsae"
    path.write_bytes(blob)

    report = verifier.verify_evidence_artifact(path, audit_keys={7: key})

    assert report.ok, report.errors
    assert report.work_certificate.full_artifact_materializations == 0
    assert report.work_certificate.bytes_read > len(blob)
    assert report.work_certificate.bytes_read < 2 * len(blob)
