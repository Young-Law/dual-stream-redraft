import hashlib
import math
from dataclasses import replace

import pytest

from dualstream.compact_evidence import (
    SCORE_TOLERANCE, _MANIFEST_V33, decode_compact_sequence,
    quantize_score, dequantize_score, verify_keyed_replay,
)
from dualstream.v210 import build_v33_artifact, verify_v33

KEY = b'contract-test-key'


def row(width=10):
    return {'chosen_id': 0, 'topk_ids': list(range(width)),
            'topk_scores': [1.0 / (i + 2) for i in range(width)]}


def artifact(**kwargs):
    return build_v33_artifact([row()], audit_key=KEY, audit_key_id=7,
                              stochastic_rate_ppm=1000000, **kwargs)


def test_keyed_verification_requires_key():
    data = artifact()
    with pytest.raises(ValueError, match='requires audit keys'):
        verify_v33(data)
    assert verify_v33(data, audit_keys={7: KEY})['outcome'] == 'LOCAL_PASS'
    verify_keyed_replay(data, {7: KEY}, tension_maps={})


@pytest.mark.parametrize('value', [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_invalid_scores_are_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        quantize_score(value)
    token = row()
    token['topk_scores'][-1] = value
    with pytest.raises(ValueError, match='finite'):
        build_v33_artifact([token], stochastic_rate_ppm=0)


def test_selected_token_requires_full_width():
    with pytest.raises(ValueError, match='requires 10 candidates'):
        build_v33_artifact([row(5)], audit_key=KEY, stochastic_rate_ppm=1000000)
    decoded = decode_compact_sequence(artifact())
    token = decoded['tokens'][0]
    decoded['tokens'][0] = replace(token, topk_ids=token.topk_ids[:5],
                                   topk_scores=token.topk_scores[:5], effective_topk=5)
    with pytest.raises(ValueError, match='insufficient candidate width'):
        verify_keyed_replay(decoded, {7: KEY})


def test_probability_metadata_and_quantization_bound():
    decoded = decode_compact_sequence(artifact())
    assert decoded['meta']['score_representation'] == 'pre-control-softmax-probability'
    assert decoded['meta']['topk_renormalized'] is False
    for i in range(10001):
        value = i / 10000
        assert abs(dequantize_score(quantize_score(value)) - value) <= SCORE_TOLERANCE + 1e-15


def test_canonical_content_preimage_includes_policy_commitment():
    policy = {'retention_days': 30}
    data = artifact(retention_requirement=policy)
    decoded = decode_compact_sequence(data)
    offset = len(data) - _MANIFEST_V33.size
    preimage = data[:offset] + bytes(32) + data[offset + 32:]
    assert decoded['manifest'].artifact_content_hash == hashlib.sha256(preimage).hexdigest()
    assert decoded['manifest'].retention_requirement_hash == hashlib.sha256(b'{"retention_days":30}').hexdigest()
