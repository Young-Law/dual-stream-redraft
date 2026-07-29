import binascii
import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dualstream.compact_evidence import (
    MAGIC,
    PREFIX,
    TRIGGER_CANARY,
    TRIGGER_HISTORY,
    TRIGGER_RANK,
    TRIGGER_STOCHASTIC,
    VERSION_V31,
    VERSION_V32,
    VERSION_V33,
    _CHUNK_V33,
    _HEADER_V31,
    _HEADER_V33,
    _MANIFEST_V33,
    _TOKEN_V33_PREFIX,
    decode_compact_sequence,
    encode_compact_sequence,
    reconstruct_token_evidence,
    verify_keyed_replay,
    SignalSpanEventV3,
)
from dualstream.generator import DualStreamGenerator, GenerationConfig
from dualstream.retention import compute_evidence_budget_summary
from dualstream.verifier import verify_evidence_artifact

KEY = b"phase1-authorized-key"
WRONG_KEY = b"phase1-wrong-key"


def rows(count=16, width=10, rank_every=0):
    out = []
    for i in range(count):
        ids = list(range(100000 + i * width, 100000 + i * width + width))
        chosen = ids[6] if rank_every and i % rank_every == 0 else ids[0]
        out.append({"chosen_id": chosen, "topk_ids": ids, "topk_scores": [1.0 - j / 20 for j in range(width)]})
    return out


def v33(**kwargs):
    return encode_compact_sequence(rows(32, 10, rank_every=7), wire_version=VERSION_V33, audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=200000, adaptive_k=True, **kwargs)


def mutate(data, offset, value):
    buf = bytearray(data)
    buf[offset] ^= value
    return bytes(buf)


def rebuild_v33_public_integrity(artifact, *, payload_mutator=None, manifest_floor=None):
    """Rebuild every unkeyed integrity field after an adversarial edit."""
    buf = bytearray(artifact)
    metadata_len = _HEADER_V33.unpack_from(buf)[25]
    chunk_offset = _HEADER_V33.size + 32 + metadata_len
    chunk = list(_CHUNK_V33.unpack_from(buf, chunk_offset))
    payload_offset = chunk_offset + _CHUNK_V33.size
    payload = bytearray(buf[payload_offset:payload_offset + chunk[10]])
    if payload_mutator is not None:
        payload_mutator(payload, chunk)
    chunk[10] = len(payload)
    chunk[11] = binascii.crc32(payload) & 0xFFFFFFFF
    chunk[12] = hashlib.sha256(payload).digest()
    buf[chunk_offset:payload_offset] = _CHUNK_V33.pack(*chunk)
    buf[payload_offset:payload_offset + len(payload)] = payload

    manifest_offset = len(buf) - _MANIFEST_V33.size
    fields = list(_MANIFEST_V33.unpack_from(buf, manifest_offset))
    fields[2] = hashlib.sha256(payload).digest()  # single-chunk Merkle root
    if manifest_floor is not None:
        fields[7] = manifest_floor
    fields[0] = b"\0" * 32
    zero_manifest = _MANIFEST_V33.pack(*fields)
    fields[0] = hashlib.sha256(bytes(buf[:manifest_offset]) + zero_manifest).digest()
    buf[manifest_offset:] = _MANIFEST_V33.pack(*fields)
    return bytes(buf)


def rebuild_v33_header_integrity(artifact, field_mutator):
    """Alter a header declaration and rebuild all unkeyed enclosing hashes."""
    buf = bytearray(artifact)
    fields = list(_HEADER_V33.unpack_from(buf))
    field_mutator(fields)
    header = _HEADER_V33.pack(*fields)
    buf[:_HEADER_V33.size] = header
    manifest_offset = len(buf) - _MANIFEST_V33.size
    manifest = list(_MANIFEST_V33.unpack_from(buf, manifest_offset))
    manifest[0] = b"\0" * 32
    manifest[1] = hashlib.sha256(header).digest()
    zero_manifest = _MANIFEST_V33.pack(*manifest)
    manifest[0] = hashlib.sha256(bytes(buf[:manifest_offset]) + zero_manifest).digest()
    buf[manifest_offset:] = _MANIFEST_V33.pack(*manifest)
    return bytes(buf)


def first_record_trigger_mutator(set_bits=0, clear_bits=0):
    def edit(payload, chunk):
        old = payload[2]
        new = (old | set_bits) & ~clear_bits
        payload[2] = new
        for field, bit in ((6, TRIGGER_RANK), (7, TRIGGER_STOCHASTIC), (8, TRIGGER_HISTORY), (9, TRIGGER_CANARY)):
            chunk[field] += bool(new & bit) - bool(old & bit)
    return edit


def test_v31_and_v32_decoder_compatibility_and_unknown_version():
    legacy_v32 = encode_compact_sequence(rows(2, 3), wire_version=VERSION_V32, adaptive_k=False)
    assert decode_compact_sequence(legacy_v32)["header"].schema_version == VERSION_V32

    header = _HEADER_V31.pack(MAGIC, VERSION_V31, len("DSA-CI-Lite"), 2, 1, 1, 1, 3, 3, 1, 0)
    body = b"{}"
    token_body = struct.pack("<IBB", 11, 0, 3) + struct.pack("<IB", 11, 255) + struct.pack("<IB", 12, 1) + struct.pack("<IB", 13, 0)
    chunk = struct.pack("<IIIII", 0, 0, 1, len(token_body), 0)
    chunk = chunk[:-4] + struct.pack("<I", __import__("binascii").crc32(token_body) & 0xFFFFFFFF)
    legacy_v31 = header + b"DSA-CI-Lite" + body + chunk + token_body
    assert decode_compact_sequence(legacy_v31)["header"].schema_version == VERSION_V31

    bad = bytearray(legacy_v32)
    bad[8:10] = (0x9999).to_bytes(2, "little")
    with pytest.raises(ValueError, match="unsupported"):
        decode_compact_sequence(bytes(bad))


def test_v33_binary_round_trip_prefix_and_no_json_discriminator():
    artifact = v33()
    assert artifact.startswith(MAGIC + struct.pack("<H", VERSION_V33))
    assert artifact[:1] != b"{"
    decoded = decode_compact_sequence(artifact)
    assert decoded["header"].schema_version == VERSION_V33
    assert decoded["header"].token_count == 32
    assert decoded["manifest"].token_count == 32
    assert decoded["tokens"][0].chosen_id == rows(1, 10, rank_every=1)[0]["chosen_id"]


def test_malformed_truncated_corrupt_and_reordered_v33_rejected():
    artifact = v33()
    with pytest.raises(ValueError):
        decode_compact_sequence(artifact[:20])
    with pytest.raises(ValueError):
        decode_compact_sequence(mutate(artifact, 30, 0xFF))

    header_size = _HEADER_V33.size
    metadata_len = struct.unpack_from("<H", artifact, header_size - 8)[0]
    chunk_offset = header_size + 32 + metadata_len
    corrupt_chunk = mutate(artifact, chunk_offset + _CHUNK_V33.size + 1, 0x01)
    with pytest.raises(ValueError, match="chunk integrity"):
        decode_compact_sequence(corrupt_chunk)
    reordered = bytearray(artifact)
    reordered[chunk_offset] = 1
    with pytest.raises(ValueError, match="reordered"):
        decode_compact_sequence(bytes(reordered))


def test_missing_duplicate_token_records_rejected():
    artifact = encode_compact_sequence(rows(2, 3), wire_version=VERSION_V33, adaptive_k=False)
    decoded = decode_compact_sequence(artifact)
    assert len(decoded["tokens"]) == 2
    header_size = _HEADER_V33.size
    metadata_len = struct.unpack_from("<H", artifact, header_size - 8)[0]
    chunk_offset = header_size + 32 + metadata_len
    missing = bytearray(artifact)
    struct.pack_into("<H", missing, chunk_offset + 8, 1)
    with pytest.raises(ValueError):
        decode_compact_sequence(bytes(missing))

    duplicate = bytearray(artifact)
    struct.pack_into("<I", duplicate, chunk_offset + 4, 1)
    with pytest.raises(ValueError):
        decode_compact_sequence(bytes(duplicate))


@pytest.mark.parametrize(
    "indexes",
    [
        [0, 0],       # duplicate
        [0, 2],       # missing index 1
        [1, 0],       # reordered
        [0, 1, 3],    # noncontiguous
    ],
)
def test_v33_encoder_rejects_invalid_token_indexes(indexes):
    source = rows(len(indexes), 3)
    for rec, token_index in zip(source, indexes):
        rec["token_index"] = token_index
    with pytest.raises(ValueError, match="does not match expected"):
        encode_compact_sequence(source, wire_version=VERSION_V33, adaptive_k=False)


def test_chosen_reconstruction_fixed_variable_and_triggers():
    artifact = v33()
    evidence = reconstruct_token_evidence(artifact)
    assert evidence[0]["chosen_id"] == rows(1, 10, rank_every=1)[0]["chosen_id"]
    assert evidence[0]["effective_topk"] == 7
    assert evidence[0]["trigger_flags"] & TRIGGER_RANK

    fixed = encode_compact_sequence(rows(3, 3), wire_version=VERSION_V33, adaptive_k=False)
    assert {r["effective_topk"] for r in reconstruct_token_evidence(fixed)} == {3}

    hist_rows = rows(2, 10)
    hist_rows[0]["trigger_flags"] = TRIGGER_HISTORY
    decoded = decode_compact_sequence(encode_compact_sequence(hist_rows, wire_version=VERSION_V33, adaptive_k=True))
    assert decoded["tokens"][0].trigger_flags & TRIGGER_HISTORY

    multi_rows = rows(1, 10, rank_every=1)
    multi_rows[0]["trigger_flags"] = TRIGGER_HISTORY
    multi = decode_compact_sequence(encode_compact_sequence(multi_rows, wire_version=VERSION_V33, adaptive_k=True))
    assert multi["tokens"][0].trigger_flags & TRIGGER_RANK
    assert multi["tokens"][0].trigger_flags & TRIGGER_HISTORY


def test_canary_separation():
    canary_rows = rows(1, 10)
    canary_rows[0]["trigger_flags"] = TRIGGER_CANARY
    with pytest.raises(ValueError, match="canary"):
        encode_compact_sequence(canary_rows, wire_version=VERSION_V33)
    artifact = encode_compact_sequence(canary_rows, wire_version=VERSION_V33, profile="DSA-Deep", canary_eval=True)
    assert decode_compact_sequence(artifact)["tokens"][0].trigger_flags & TRIGGER_CANARY


def test_exact_keyed_replay_and_tamper_rejection():
    artifact = v33()
    decoded = decode_compact_sequence(artifact)
    verify_keyed_replay(decoded, {3: KEY})
    with pytest.raises(ValueError, match="commitment|replay"):
        verify_keyed_replay(decoded, {3: WRONG_KEY})
    with pytest.raises(ValueError, match="unknown audit key"):
        verify_keyed_replay(decoded, {99: KEY})
    assert KEY not in artifact

    # Alter commitment-bound public replay context.
    tampered = copy.deepcopy(decoded)
    tampered["meta"]["benchmark_id"] = "changed"
    with pytest.raises(ValueError, match="commitment"):
        verify_keyed_replay(tampered, {3: KEY})

    # Added or removed stochastic trigger flags must fail replay.
    changed = copy.deepcopy(decoded)
    for token in changed["tokens"]:
        if not token.trigger_flags & (TRIGGER_RANK | TRIGGER_HISTORY | TRIGGER_CANARY):
            changed_token = token
128 continuation truncated...