import pytest
from dualstream.compact_evidence import encode_compact_sequence, decode_compact_sequence, reconstruct_token_evidence, SignalSpanEventV3


def rows(n=4):
    return [{"token_index": i, "chosen_id": 100+i, "topk_ids": [100+i, 200+i, 300+i], "topk_scores": [0.6, 0.3, 0.1]} for i in range(n)]


def test_compact_round_trip_reconstructs_tokens_and_spans():
    blob = encode_compact_sequence(rows(), spans=[SignalSpanEventV3(1, 3, 7, 0.5)])
    recs = reconstruct_token_evidence(blob)
    assert [r["chosen_id"] for r in recs] == [100, 101, 102, 103]
    assert recs[0]["topk_ids"] == [100, 200, 300]
    assert abs(recs[0]["topk_scores"][0] - 0.6) <= 1 / 255
    assert [len(r["signals"]) for r in recs] == [0, 1, 1, 0]


def test_corrupted_chunk_fails():
    data = bytearray(encode_compact_sequence(rows()))
    data[-4] ^= 0xFF
    with pytest.raises(ValueError):
        decode_compact_sequence(bytes(data))


def test_truncated_artifact_fails():
    with pytest.raises(ValueError):
        decode_compact_sequence(encode_compact_sequence(rows())[:-2])


def test_reordered_token_evidence_fails_with_valid_crc():
    from dualstream import compact_evidence as ce
    data = bytearray(encode_compact_sequence(rows(3)))
    header_size = ce._HEADER.size
    magic, version, profile_len, meta_len, seq, token_count, cap, base_k, max_k, chunk_count, span_count = ce._HEADER.unpack_from(data, 0)
    pos = header_size + profile_len + meta_len
    chunk_index, start, count, byte_len, crc = ce._CHUNK.unpack_from(data, pos)
    body_start = pos + ce._CHUNK.size
    token_size = ce._TOKEN.size + 3 * ce._TOPK.size
    first = data[body_start:body_start + token_size]
    second = data[body_start + token_size:body_start + 2 * token_size]
    data[body_start:body_start + token_size] = second
    data[body_start + token_size:body_start + 2 * token_size] = first
    new_crc = ce.binascii.crc32(data[body_start:body_start + byte_len]) & 0xFFFFFFFF
    ce._CHUNK.pack_into(data, pos, chunk_index, start, count, byte_len, new_crc)
    with pytest.raises(ValueError, match="missing, duplicate, or reordered"):
        decode_compact_sequence(bytes(data))


def test_duplicate_token_evidence_fails_with_valid_crc():
    from dualstream import compact_evidence as ce
    data = bytearray(encode_compact_sequence(rows(3)))
    header_size = ce._HEADER.size
    magic, version, profile_len, meta_len, seq, token_count, cap, base_k, max_k, chunk_count, span_count = ce._HEADER.unpack_from(data, 0)
    pos = header_size + profile_len + meta_len
    chunk_index, start, count, byte_len, crc = ce._CHUNK.unpack_from(data, pos)
    body_start = pos + ce._CHUNK.size
    token_size = ce._TOKEN.size + 3 * ce._TOPK.size
    first = data[body_start:body_start + token_size]
    data[body_start + token_size:body_start + 2 * token_size] = first
    new_crc = ce.binascii.crc32(data[body_start:body_start + byte_len]) & 0xFFFFFFFF
    ce._CHUNK.pack_into(data, pos, chunk_index, start, count, byte_len, new_crc)
    with pytest.raises(ValueError, match="missing, duplicate, or reordered"):
        decode_compact_sequence(bytes(data))


def _legacy_v31_blob():
    from dualstream import compact_evidence as ce
    import binascii, json
    profile = b"DSA-CI-Lite"
    meta = json.dumps({"profile_id": "DSA-CI-Lite", "verifier_budget_id": "h3e-ci-lite"}, sort_keys=True, separators=(",", ":")).encode()
    rows_ = rows(2)
    body = bytearray()
    for offset, r in enumerate(rows_):
        body += ce._TOKEN.pack(r["chosen_id"], offset, 3)
        for tid, score in zip(r["topk_ids"], r["topk_scores"]):
            body += ce._TOPK.pack(tid, ce.quantize_score(score))
    crc = binascii.crc32(body) & 0xFFFFFFFF
    chunk = ce._CHUNK.pack(0, 0, 2, len(body), crc) + body
    header = ce._HEADER_V31.pack(ce.MAGIC, ce.VERSION_V31, len(profile), len(meta), 0, 2, 256, 3, 5, 1, 0)
    return header + profile + meta + chunk


def test_v31_legacy_artifact_decodes_after_v32_layout_change():
    recs = reconstruct_token_evidence(_legacy_v31_blob())
    assert [r["chosen_id"] for r in recs] == [100, 101]
    assert recs[0]["chosen_rank"] == 1


def test_v32_artifact_decodes_with_new_layout():
    blob = encode_compact_sequence(rows(2))
    decoded = decode_compact_sequence(blob)
    assert decoded["header"].schema_version == 0x0302
    assert [r["chosen_id"] for r in reconstruct_token_evidence(decoded)] == [100, 101]


def test_layout_mismatch_reports_distinct_error():
    bad = bytearray(encode_compact_sequence(rows(1)))
    from dualstream import compact_evidence as ce
    ce._PREFIX.pack_into(bad, 0, ce.MAGIC, ce.VERSION_V31)
    with pytest.raises(ValueError, match="layout|version|header") as err:
        decode_compact_sequence(bytes(bad))
    assert "unknown profile" not in str(err.value)


def test_unsupported_compact_version_fails_clearly():
    bad = bytearray(encode_compact_sequence(rows(1)))
    from dualstream import compact_evidence as ce
    ce._PREFIX.pack_into(bad, 0, ce.MAGIC, 0x9999)
    with pytest.raises(ValueError, match="unsupported compact evidence version"):
        decode_compact_sequence(bytes(bad))
