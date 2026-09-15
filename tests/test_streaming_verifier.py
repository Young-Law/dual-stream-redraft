from pathlib import Path

import pytest

from dualstream import compact_evidence as ce
from dualstream.streaming_verifier import verify_stream
from dualstream.v210 import build_v33_artifact
from dualstream.verifier import verify_evidence_artifact

KEY = b'streaming-test-key'


def artifact(n=70, **kwargs):
    return build_v33_artifact([
        {'chosen_id': 1, 'topk_ids': list(range(1, 11)),
         'topk_scores': [.5 / i for i in range(1, 11)]}
        for _ in range(n)
    ], audit_key=KEY, audit_key_id=7, stochastic_rate_ppm=1000000,
       chunk_token_capacity=16, **kwargs)


def test_streaming_matches_decoder_without_materialization(tmp_path, monkeypatch):
    data = artifact()
    expected = ce.decode_compact_sequence(data)
    path = tmp_path / 'compact_evidence.dsae'
    path.write_bytes(data)
    def forbidden(*args, **kwargs):
        raise AssertionError('full materialization attempted')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    monkeypatch.setattr(ce, 'decode_compact_sequence', forbidden)
    result = verify_stream(path, {7: KEY})
    assert result['manifest'] == expected['manifest']
    assert result['sha256'] == expected['sha256']
    assert 'tokens' not in result
    counters = result['counters']
    assert counters['full_artifact_materializations'] == 0
    assert counters['replay_passes'] == 4
    assert counters['token_records_decoded'] == 70 * 5
    assert counters['candidate_entries_decoded'] == 700 * 5
    report = verify_evidence_artifact(path, audit_keys={7: KEY}, enforce_budget=False)
    assert report.ok, report.errors
    assert report.work_certificate.work_bounds['max_metadata_bytes'] == 1024 * 1024


def test_failed_replay_preserves_work_and_cannot_pass_retention(tmp_path):
    path = tmp_path / 'compact_evidence.dsae'
    path.write_bytes(artifact())
    report = verify_evidence_artifact(path, enforce_budget=False)
    assert not report.ok
    assert report.retention_state == 'NOT_VERIFIED'
    assert report.work_certificate.token_records_decoded == 70
    assert report.work_certificate.bytes_read > 0
    assert report.work_certificate.full_artifact_materializations == 0


def test_metadata_limit_precedes_read(tmp_path):
    data = artifact()
    fields = list(ce._HEADER_V33.unpack_from(data))
    fields[-4] = 65535
    path = tmp_path / 'compact_evidence.dsae'
    path.write_bytes(ce._HEADER_V33.pack(*fields))
    counters = {}
    with pytest.raises(ValueError, match='work bound.*metadata'):
        verify_stream(path, counters=counters, max_metadata_bytes=1024)
    assert counters['bytes_read'] == ce._HEADER_V33.size
    fields[-8] = 1025  # chunk_token_capacity
    path.write_bytes(ce._HEADER_V33.pack(*fields))
    report = verify_evidence_artifact(path)
    assert 523 in report.failure_codes
    assert report.work_certificate.bytes_read == ce._HEADER_V33.size + ce.PREFIX.size


@pytest.mark.parametrize('n', [1, 16, 17, 48, 80, 112])
def test_streaming_merkle_frontier_matches_wire_root(tmp_path, n):
    path = tmp_path / 'compact_evidence.dsae'
    path.write_bytes(artifact(n))
    assert verify_stream(path, {7: KEY})['stats']['token_count'] == n


def test_signature_binds_artifact_and_ignores_runtime_observations(tmp_path):
    from dataclasses import replace
    from dualstream.verifier import verify_work_certificate_signature
    path = tmp_path / 'compact_evidence.dsae'
    path.write_bytes(artifact())
    report = verify_evidence_artifact(path, audit_keys={7: KEY},
                                      enforce_budget=False, verifier_key=KEY)
    cert = report.work_certificate
    assert cert.work_completed
    changed_host = replace(report, runtime_diagnostics=replace(
        report.runtime_diagnostics, elapsed_seconds=9999, peak_tracemalloc_bytes=999999999))
    assert verify_work_certificate_signature(changed_host.work_certificate, cert.signature, KEY)
    for changed in (replace(cert, artifact_sha256='0' * 64),
                    replace(cert, work_completed=False),
                    replace(cert, maximum_live_bytes=999999999)):
        assert not verify_work_certificate_signature(changed, cert.signature, KEY)
