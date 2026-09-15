import hashlib
import hmac
import time
import zlib
from dataclasses import replace

import pytest

from dualstream.challenge import (ChallengeResponse, issue_possession_challenge,
                                  respond_to_challenge, verify_possession_challenge)
from dualstream.compact_evidence import encode_compact_sequence
from dualstream.retention_manager import RetentionPipeline, PipelineStatus
from dualstream.storage_validator import LocalFilesystemBackend
from dualstream.migration import AllowedTransform, verify_after_transform
from dualstream.retention_lifecycle import verify_receipt_chain


def blob():
    return encode_compact_sequence([{"token_index": 0, "chosen_id": 100,
        "topk_ids": [100, 200, 300], "topk_scores": [0.6, 0.3, 0.1]}])


def pipeline(tmp_path, backend=None):
    return RetentionPipeline(backend or LocalFilesystemBackend(tmp_path), b'issuer', b'validator', b'challenger')


def run(p, data=None, **kwargs):
    return p.run_full(artifact_id='artifact', artifact_bytes=blob() if data is None else data,
                      issuer_id='issuer', validator_id='validator', **kwargs)


def test_public_digest_echo_and_unsigned_response_fail():
    c = issue_possession_challenge('artifact', b'secret evidence bytes', b'challenger')
    forged = ChallengeResponse(c.challenge_id, 'holder', c.expected_hash, nonce=c.nonce)
    assert not verify_possession_challenge(c, forged, b'challenger', b'holder')['valid']
    # Even a legitimate holder key cannot replace the bytes with an echoed digest.
    forged.signature = hmac.digest(b'holder', forged.to_canonical_json().encode(), 'sha256')
    assert not verify_possession_challenge(c, forged, b'challenger', b'holder')['valid']
    response = respond_to_challenge(c, b'secret evidence bytes', 'holder', b'holder')
    assert verify_possession_challenge(c, response, b'challenger', b'holder', 'holder')['valid']
    assert not verify_possession_challenge(c, replace(response, signature=b''), b'challenger', b'holder')['valid']
    assert not hasattr(c, 'challenge_key')


def test_challenge_identity_nonce_and_range_enforced():
    c = issue_possession_challenge('a', b'evidence', b'key')
    r = respond_to_challenge(c, b'evidence', 'holder', b'holder')
    assert not verify_possession_challenge(c, r, b'key', b'holder', 'other')['valid']
    r.nonce = 'other'
    r.signature = hmac.digest(b'holder', r.to_canonical_json().encode(), 'sha256')
    assert not verify_possession_challenge(c, r, b'key', b'holder')['valid']
    with pytest.raises(ValueError):
        issue_possession_challenge('a', b'', b'key')


def test_invalid_evidence_bounds_and_horizon_fail_before_storage(tmp_path):
    p = pipeline(tmp_path)
    for kwargs in ({'data': b'ordinary text'}, {'max_artifact_bytes': 1},
                   {'min_retention_days': 0}, {'expires_at': time.time() + 60},
                   {'profile_name': 'wrong-profile'}):
        assert run(p, **kwargs).overall_status == PipelineStatus.FAILED
        assert not p.storage.exists('artifact')


def test_pipeline_success_is_explicitly_local_and_binds_object(tmp_path):
    p = pipeline(tmp_path)
    result = run(p)
    assert result.overall_status == PipelineStatus.COMPLETED
    assert result.to_dict()['independent_retention_attested'] is False
    assert result.to_dict()['assurance_scope'] == 'local_research_validation'
    assert result.requirement.artifact_content_hash == hashlib.sha256(blob()).digest()
    receipt = replace(result.receipt, artifact_id='other')
    receipt.signature = hmac.digest(b'validator', receipt.to_canonical_json().encode(), 'sha256')
    assert not verify_receipt_chain(result.requirement, receipt, blob(), b'issuer', b'validator')['valid']
    receipt = replace(result.receipt, retain_until=0)
    receipt.signature = hmac.digest(b'validator', receipt.to_canonical_json().encode(), 'sha256')
    assert not verify_receipt_chain(result.requirement, receipt, blob(), b'issuer', b'validator')['valid']


def test_missing_storage_cannot_be_answered_from_original_memory(tmp_path):
    p = pipeline(tmp_path)
    assert run(p).overall_status == PipelineStatus.COMPLETED
    p.storage.delete('artifact')
    assert not p.run_possession_audit(artifact_id='artifact', artifact_bytes=blob())['all_passed']
    assert not p.run_possession_audit(artifact_id='artifact', artifact_bytes=blob(), num_challenges=0)['all_passed']


def test_role_keys_required(tmp_path):
    with pytest.raises(ValueError):
        RetentionPipeline(LocalFilesystemBackend(tmp_path))
    with pytest.raises(ValueError):
        RetentionPipeline(LocalFilesystemBackend(tmp_path), b'same', b'same', b'other')


def test_restore_requires_reconstruction_not_small_or_nonempty_bytes():
    data = blob()
    assert verify_after_transform(data, zlib.compress(data), AllowedTransform.TRANSFORM_COMPRESS).valid
    for transform in AllowedTransform:
        assert not verify_after_transform(data, b'x', transform).valid
    assert not verify_after_transform(b'text', b'text', AllowedTransform.TRANSFORM_REPLICATE).valid


def test_full_pipeline_fetches_storage_again_for_challenge(tmp_path):
    class EvaporatingBackend(LocalFilesystemBackend):
        reads = 0

        def retrieve(self, artifact_id):
            self.reads += 1
            # Storage validation and receipt see the object; possession does not.
            if self.reads >= 3:
                return None
            return super().retrieve(artifact_id)

    result = run(pipeline(tmp_path, EvaporatingBackend(tmp_path)))
    assert result.overall_status != PipelineStatus.COMPLETED
    assert any(s.step.value == 'challenge_verified' and not s.ok for s in result.steps)
