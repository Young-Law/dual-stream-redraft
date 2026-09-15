from dataclasses import replace

import pytest

from dualstream import compact_evidence as ce
from dualstream.streaming_verifier import verify_stream
from dualstream.v210 import build_v33_artifact


KEY = b"post-merge-streaming-p1-key"


def artifact(n=8):
    return build_v33_artifact(
        [
            {
                "chosen_id": 1,
                "topk_ids": list(range(1, 11)),
                "topk_scores": [0.5 / i for i in range(1, 11)],
            }
            for _ in range(n)
        ],
        audit_key=KEY,
        audit_key_id=7,
        stochastic_rate_ppm=1000000,
        chunk_token_capacity=8,
    )


def mutate_header(data, index, value):
    fields = list(ce._HEADER_V33.unpack_from(data))
    fields[index] = value
    return ce._HEADER_V33.pack(*fields) + data[ce._HEADER_V33.size :]


@pytest.mark.parametrize(
    ("field_index", "bad_value", "message"),
    [
        (20, 2, "unsupported V3.3 quantization identifier"),
        (22, 2, "unsupported V3.3 verifier work-profile identifier"),
    ],
)
def test_streaming_rejects_unsupported_semantic_header_ids(
    tmp_path, field_index, bad_value, message
):
    path = tmp_path / "unsupported-header-id.dsae"
    path.write_bytes(mutate_header(artifact(), field_index, bad_value))

    with pytest.raises(ValueError, match=message):
        verify_stream(path, {7: KEY})


def test_streaming_rejects_duplicate_candidate_ids(tmp_path, monkeypatch):
    path = tmp_path / "duplicate-candidates.dsae"
    path.write_bytes(artifact())

    original_decode = ce._decode_v33_chunk_payload

    def duplicate_first_candidate(payload, start, count, base):
        records = original_decode(payload, start, count, base)
        first = records[0]
        duplicate_ids = (first.topk_ids[0], first.topk_ids[0], *first.topk_ids[2:])
        records[0] = replace(first, topk_ids=duplicate_ids)
        return records

    monkeypatch.setattr(ce, "_decode_v33_chunk_payload", duplicate_first_candidate)

    with pytest.raises(ValueError, match="duplicate V3.3 candidate token IDs"):
        verify_stream(path, {7: KEY})
