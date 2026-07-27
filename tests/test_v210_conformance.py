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

    header_size = 197
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
    header_size = 197
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
            break
    changed["tokens"][changed_token.token_index] = changed_token.__class__(
        changed_token.token_index,
        changed_token.chosen_id,
        changed_token.topk_ids,
        changed_token.topk_scores,
        changed_token.effective_topk,
        changed_token.chosen_rank,
        changed_token.trigger_flags ^ TRIGGER_STOCHASTIC,
        changed_token.record_flags,
    )
    with pytest.raises(ValueError, match="replay mismatch"):
        verify_keyed_replay(changed, {3: KEY})


def test_generator_preserves_full_64_bit_prompt_nonce(monkeypatch):
    nonce = 0x12345678ABCDEF01
    monkeypatch.setattr("dualstream.generator.random.getrandbits", lambda _: nonce)
    gen = DualStreamGenerator.__new__(DualStreamGenerator)
    gen.model_name, gen.device = "fake", "cpu"
    gen.tokenizer, gen.model = FakeTokenizer(), FakeModel()
    cfg = GenerationConfig(max_new_tokens=2, compact_evidence=True, compact_wire_version=VERSION_V33,
                           adaptive_k=True, top_k=10, do_sample=False, audit_key=KEY,
                           audit_key_id=3, stochastic_rate_ppm=1_000_000)
    decoded = decode_compact_sequence(gen.generate("prompt", cfg)["compact_evidence_bytes"])
    assert decoded["header"].sequence_id == nonce & 0xFFFFFFFF
    assert decoded["meta"]["prompt_nonce"] == nonce
    verify_keyed_replay(decoded, {3: KEY})
    truncated = copy.deepcopy(decoded)
    truncated["meta"]["prompt_nonce"] = decoded["header"].sequence_id
    with pytest.raises(ValueError, match="commitment"):
        verify_keyed_replay(truncated, {3: KEY})


def test_replay_recomputes_commit_identity_after_public_hash_rebuild():
    artifact = encode_compact_sequence(rows(8, 10), wire_version=VERSION_V33, audit_key=KEY,
                                       audit_key_id=3, stochastic_rate_ppm=500000)
    def alter_candidate(payload, _chunk):
        # First token prefix is four bytes; alter its first retained token id.
        struct.pack_into("<I", payload, _TOKEN_V33_PREFIX.size,
                         struct.unpack_from("<I", payload, _TOKEN_V33_PREFIX.size)[0] + 1)
    tampered = rebuild_v33_public_integrity(artifact, payload_mutator=alter_candidate)
    decoded = decode_compact_sequence(tampered)  # all public integrity checks pass
    with pytest.raises(ValueError, match="commit identity"):
        verify_keyed_replay(decoded, {3: KEY})


@pytest.mark.parametrize("exemption", [TRIGGER_HISTORY, TRIGGER_CANARY])
def test_forged_history_or_canary_exemption_rejected(exemption):
    artifact = encode_compact_sequence(rows(1, 10), wire_version=VERSION_V33, audit_key=KEY,
                                       audit_key_id=3, stochastic_rate_ppm=1_000_000)
    tampered = rebuild_v33_public_integrity(
        artifact, payload_mutator=first_record_trigger_mutator(exemption, TRIGGER_STOCHASTIC))
    with pytest.raises(ValueError, match="eligibility|commitment"):
        verify_keyed_replay(tampered, {3: KEY})


def test_false_rank_exemption_and_changed_legitimate_eligibility_rejected():
    artifact = encode_compact_sequence(rows(1, 10), wire_version=VERSION_V33, audit_key=KEY,
                                       audit_key_id=3, stochastic_rate_ppm=1_000_000)
    false_rank = rebuild_v33_public_integrity(
        artifact, payload_mutator=first_record_trigger_mutator(TRIGGER_RANK, TRIGGER_STOCHASTIC))
    with pytest.raises(ValueError, match="rank eligibility"):
        verify_keyed_replay(false_rank, {3: KEY})

    history_rows = rows(1, 10)
    history_rows[0]["trigger_flags"] = TRIGGER_HISTORY
    legitimate = encode_compact_sequence(history_rows, wire_version=VERSION_V33, audit_key=KEY,
                                          audit_key_id=3, stochastic_rate_ppm=1_000_000)
    verify_keyed_replay(legitimate, {3: KEY})
    changed = rebuild_v33_public_integrity(
        legitimate, payload_mutator=first_record_trigger_mutator(TRIGGER_CANARY, TRIGGER_HISTORY))
    with pytest.raises(ValueError, match="eligibility|commitment"):
        verify_keyed_replay(changed, {3: KEY})


def test_profile_and_maximum_k_substitution_rejected_after_public_hash_rebuild():
    artifact = encode_compact_sequence(
        rows(4, 10), wire_version=VERSION_V33, profile="DSA-CI-Standard",
        audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=500_000,
    )
    # Deep and CI-Standard share base K=5, so this remains structurally valid and
    # specifically exercises the keyed profile binding rather than the decoder's
    # ordinary profile/base-K consistency check.
    changed_profile = rebuild_v33_header_integrity(
        artifact, lambda fields: fields.__setitem__(2, 3)
    )
    with pytest.raises(ValueError, match="profile|commitment"):
        verify_keyed_replay(changed_profile, {3: KEY})

    changed_max_k = rebuild_v33_header_integrity(
        artifact, lambda fields: fields.__setitem__(13, fields[13] + 1)
    )
    decode_compact_sequence(changed_max_k)
    with pytest.raises(ValueError, match="commitment"):
        verify_keyed_replay(changed_max_k, {3: KEY})


def test_authenticated_multi_trigger_record_replays():
    multi_rows = rows(1, 10, rank_every=1)
    multi_rows[0]["trigger_flags"] = TRIGGER_HISTORY | TRIGGER_CANARY
    artifact = encode_compact_sequence(multi_rows, wire_version=VERSION_V33, audit_key=KEY,
                                       audit_key_id=3, stochastic_rate_ppm=1_000_000,
                                       canary_eval=True)
    decoded = decode_compact_sequence(artifact)
    assert decoded["tokens"][0].trigger_flags & TRIGGER_RANK
    verify_keyed_replay(decoded, {3: KEY})


def test_v33_retention_floor_is_exact_and_manifest_manipulation_fails():
    special = rows(3, 10, rank_every=2)
    special[-1]["chosen_id"] = 999999  # fallback chosen-id encoding
    spans = [SignalSpanEventV3(0, 1, 7, .5, 11), SignalSpanEventV3(1, 3, 8, .7, 12, 99)]
    artifact = encode_compact_sequence(special, wire_version=VERSION_V33, spans=spans,
                                       audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=1)
    decoded = decode_compact_sequence(artifact)
    assert decoded["manifest"].minimum_reconstructable_bytes == len(artifact)
    assert compute_evidence_budget_summary(artifact).minimum_reconstructable_bytes == len(artifact)
    for declared in (len(artifact) - 1, len(artifact) + 1):
        tampered = rebuild_v33_public_integrity(artifact, manifest_floor=declared)
        with pytest.raises(ValueError, match="minimum reconstructable"):
            decode_compact_sequence(tampered)


def test_canonical_preimage_stability_and_hash_tampering():
    artifact1 = v33(sequence_id=123)
    artifact2 = v33(sequence_id=123)
    assert artifact1 == artifact2
    decoded = decode_compact_sequence(artifact1)
    assert len(decoded["manifest"].artifact_content_hash) == 64
    manifest_start = len(artifact1) - _MANIFEST_V33.size
    tampered_hash = mutate(artifact1, manifest_start, 0x01)
    with pytest.raises(ValueError, match="content hash"):
        decode_compact_sequence(tampered_hash)
    tampered_content = mutate(artifact1, 260, 0x01)
    with pytest.raises(ValueError):
        decode_compact_sequence(tampered_content)


def test_shared_consumers_accept_v33(tmp_path):
    artifact = encode_compact_sequence(rows(10000, 3), wire_version=VERSION_V33, profile="DSA-CI-Lite", adaptive_k=False)
    path = tmp_path / "artifact.dsae"
    path.write_bytes(artifact)
    assert reconstruct_token_evidence(artifact)[0]["chosen_rank"] == 1
    summary = compute_evidence_budget_summary(artifact, "DSA-CI-Lite")
    assert summary.raw_bytes_per_token <= 24
    report = verify_evidence_artifact(path, profile="DSA-CI-Lite", ci_mode="pr")
    assert report.ok, report.errors
    cp = subprocess.run(
        [sys.executable, "-m", "dualstream.cli", "verify-evidence-budget", "--artifact", str(path), "--profile", "DSA-CI-Lite", "--ci-mode", "pr", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr


class FakeTokenizer:
    eos_token_id = None
    pad_token_id = None
    bos_token_id = None
    additional_special_tokens_ids = []

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1]])}

    def decode(self, ids, skip_special_tokens=False):
        return "x"


class FakeModel:
    def __call__(self, **kwargs):
        class Output:
            pass
        out = Output()
        logits = torch.arange(0, 32, dtype=torch.float32).reshape(1, 1, 32)
        out.logits = logits
        out.past_key_values = None
        out.attentions = None
        out.hidden_states = None
        return out


def test_generator_produces_v33_compact_evidence():
    gen = DualStreamGenerator.__new__(DualStreamGenerator)
    gen.model_name = "fake"
    gen.device = "cpu"
    gen.tokenizer = FakeTokenizer()
    gen.model = FakeModel()
    cfg = GenerationConfig(max_new_tokens=2, compact_evidence=True, compact_wire_version=VERSION_V33, adaptive_k=True, top_k=10, do_sample=False)
    result = gen.generate("prompt", cfg)
    decoded = decode_compact_sequence(result["compact_evidence_bytes"])
    assert decoded["header"].schema_version == VERSION_V33
    assert decoded["header"].token_count == 2


def test_10000_token_raw_byte_ceilings():
    lite = encode_compact_sequence(rows(10000, 3), wire_version=VERSION_V33, profile="DSA-CI-Lite", adaptive_k=False)
    standard = encode_compact_sequence(rows(10000, 5), wire_version=VERSION_V33, profile="DSA-CI-Standard", adaptive_k=False)
    lite_bpt = len(lite) / 10000
    standard_bpt = len(standard) / 10000
    assert lite_bpt <= 24
    assert standard_bpt <= 48


def test_keyed_replay_preserves_full_64_bit_prompt_nonce():
    nonce = 0x1_0000_0001
    artifact = encode_compact_sequence(rows(32, 10), wire_version=VERSION_V33, audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=1_000_000, sequence_id=nonce)
    decoded = decode_compact_sequence(artifact)
    assert decoded["header"].sequence_id == (nonce & 0xFFFFFFFF)
    assert decoded["meta"]["prompt_nonce"] == nonce
    verify_keyed_replay(decoded, {3: KEY})
    truncated = copy.deepcopy(decoded)
    truncated["meta"]["prompt_nonce"] = decoded["header"].sequence_id
    with pytest.raises(ValueError, match="commitment|replay"):
        verify_keyed_replay(truncated, {3: KEY})


def test_keyed_replay_recomputes_commit_identity_from_token_evidence():
    artifact = v33(sequence_id=44)
    decoded = decode_compact_sequence(artifact)
    tampered = copy.deepcopy(decoded)
    rec = tampered["tokens"][0]
    ids = list(rec.topk_ids)
    ids[0] += 999
    tampered["tokens"][0] = rec.__class__(rec.token_index, rec.chosen_id, tuple(ids), rec.topk_scores, rec.effective_topk, rec.chosen_rank, rec.trigger_flags, rec.record_flags)
    with pytest.raises(ValueError, match="commit identity"):
        verify_keyed_replay(tampered, {3: KEY})


def test_keyed_replay_rejects_forged_stochastic_exemptions_and_preserves_multi_trigger():
    artifact = encode_compact_sequence(rows(32, 10), wire_version=VERSION_V33, audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=1_000_000, sequence_id=55)
    decoded = decode_compact_sequence(artifact)
    verify_keyed_replay(decoded, {3: KEY})
    selected = next(t for t in decoded["tokens"] if t.trigger_flags & TRIGGER_STOCHASTIC and not t.trigger_flags & TRIGGER_RANK)
    for forged in (TRIGGER_HISTORY, TRIGGER_CANARY, TRIGGER_RANK):
        tampered = copy.deepcopy(decoded)
        tampered["tokens"][selected.token_index] = selected.__class__(selected.token_index, selected.chosen_id, selected.topk_ids, selected.topk_scores, selected.effective_topk, selected.chosen_rank, (selected.trigger_flags & ~TRIGGER_STOCHASTIC) | forged, selected.record_flags)
        with pytest.raises(ValueError, match="eligibility|replay"):
            verify_keyed_replay(tampered, {3: KEY})

    hist_rows = rows(8, 10, rank_every=1)
    hist_rows[0]["trigger_flags"] = TRIGGER_HISTORY
    valid_multi = decode_compact_sequence(encode_compact_sequence(hist_rows, wire_version=VERSION_V33, audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=1_000_000, sequence_id=56))
    assert valid_multi["tokens"][0].trigger_flags & TRIGGER_RANK and valid_multi["tokens"][0].trigger_flags & TRIGGER_HISTORY
    verify_keyed_replay(valid_multi, {3: KEY})
    bad_multi = copy.deepcopy(valid_multi)
    rec = bad_multi["tokens"][0]
    bad_multi["tokens"][0] = rec.__class__(rec.token_index, rec.chosen_id, rec.topk_ids, rec.topk_scores, rec.effective_topk, rec.chosen_rank, rec.trigger_flags & ~TRIGGER_HISTORY, rec.record_flags)
    with pytest.raises(ValueError, match="eligibility"):
        verify_keyed_replay(bad_multi, {3: KEY})


def _rewrite_manifest_floor(artifact: bytes, floor: int) -> bytes:
    start = len(artifact) - _MANIFEST_V33.size
    fields = list(_MANIFEST_V33.unpack_from(artifact, start))
    fields[0] = b"\0" * 32
    fields[7] = floor
    manifest_zero = _MANIFEST_V33.pack(*fields)
    import hashlib
    fields[0] = hashlib.sha256(artifact[:start] + manifest_zero).digest()
    return artifact[:start] + _MANIFEST_V33.pack(*fields)


def _rewrite_v33_public_context(artifact: bytes, *, header_updates=None, metadata_updates=None) -> bytes:
    """Rebuild every unkeyed enclosing digest after a header/metadata edit."""
    header = list(_HEADER_V33.unpack_from(artifact, 0))
    old_metadata_len = header[25]
    metadata_start = _HEADER_V33.size + 32
    metadata = json.loads(artifact[metadata_start:metadata_start + old_metadata_len])
    for index, value in (header_updates or {}).items():
        header[index] = value
    metadata.update(metadata_updates or {})
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    header[25] = len(metadata_bytes)
    header_bytes = _HEADER_V33.pack(*header)
    old_manifest_start = len(artifact) - _MANIFEST_V33.size
    variable_body = artifact[metadata_start + old_metadata_len:old_manifest_start]
    body = header_bytes + hashlib.sha256(metadata_bytes).digest() + metadata_bytes + variable_body
    manifest = list(_MANIFEST_V33.unpack_from(artifact, old_manifest_start))
    manifest[0] = b"\0" * 32
    manifest[1] = hashlib.sha256(header_bytes).digest()
    manifest[6] = len(body) + _MANIFEST_V33.size
    manifest[7] += len(metadata_bytes) - old_metadata_len
    zero_manifest = _MANIFEST_V33.pack(*manifest)
    manifest[0] = hashlib.sha256(body + zero_manifest).digest()
    return body + _MANIFEST_V33.pack(*manifest)


def test_keyed_context_rejects_key_id_and_local_sequence_substitution():
    artifact = v33(sequence_id=0x1_0000_0009)
    key_id_changed = _rewrite_v33_public_context(artifact, header_updates={16: 4})
    assert decode_compact_sequence(key_id_changed)["meta"]["audit_key_id"] == 4
    with pytest.raises(ValueError, match="commitment"):
        verify_keyed_replay(key_id_changed, {3: KEY, 4: KEY})

    sequence_changed = _rewrite_v33_public_context(artifact, header_updates={5: 10})
    assert decode_compact_sequence(sequence_changed)["header"].sequence_id == 10
    with pytest.raises(ValueError, match="sequence id"):
        verify_keyed_replay(sequence_changed, {3: KEY})


@pytest.mark.parametrize(
    ("adaptive", "header_policy", "header_max", "metadata_policy"),
    [
        (False, 2101, 10, "dsa-v2.10-hybrid-rank-adaptive-phase1"),
        (True, 2100, 3, "dsa-v2.10-fixed-base-k-phase1"),
    ],
)
def test_keyed_context_rejects_fixed_adaptive_policy_substitution(adaptive, header_policy, header_max, metadata_policy):
    # Keep the rewritten fixed-policy case structurally decodable so the test
    # reaches authenticated policy validation rather than failing on payload size.
    width = 3 if adaptive else 10
    artifact = encode_compact_sequence(rows(32, width), wire_version=VERSION_V33, adaptive_k=adaptive, audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=200000)
    changed = _rewrite_v33_public_context(artifact, header_updates={13: header_max, 14: header_policy}, metadata_updates={"adaptive_policy": metadata_policy})
    assert decode_compact_sequence(changed)["meta"]["adaptive_policy"] == metadata_policy
    with pytest.raises(ValueError, match="eligibility|commitment|policy"):
        verify_keyed_replay(changed, {3: KEY})


def test_adaptive_policy_rejects_nonadaptive_maximum_boundary():
    with pytest.raises(ValueError, match="adaptive_k requires"):
        encode_compact_sequence(
            rows(4, 3),
            wire_version=VERSION_V33,
            profile="DSA-CI-Lite",
            adaptive_k=True,
            max_adaptive_k=3,
        )


def test_authenticated_metadata_rejects_noncanonical_json_types():
    artifact = v33(sequence_id=77)
    for field, value, message in (
        ("benchmark_id", 123, "benchmark_id must be a string"),
        ("canary_eval", "false", "canary_eval must be a boolean"),
        ("prompt_nonce", "77", "prompt_nonce must be an integer"),
        ("commit_identity", 123, "commit_identity must be a string"),
        ("pre_stochastic_eligibility_digest", False, "pre_stochastic_eligibility_digest must be a string"),
        ("adaptive_policy", 2101, "adaptive_policy must be a string"),
    ):
        changed = _rewrite_v33_public_context(artifact, metadata_updates={field: value})
        assert decode_compact_sequence(changed)["meta"][field] == value
        with pytest.raises(ValueError, match=message):
            verify_keyed_replay(changed, {3: KEY})


def test_profile_substitution_reaches_keyed_commitment_check():
    artifact = encode_compact_sequence(rows(32, 5), wire_version=VERSION_V33, profile="DSA-CI-Standard", adaptive_k=False, audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=200000)
    changed = _rewrite_v33_public_context(artifact, header_updates={2: 3}, metadata_updates={"profile_id": "DSA-Deep"})
    decoded = decode_compact_sequence(changed)
    assert decoded["header"].profile_id == decoded["meta"]["profile_id"] == "DSA-Deep"
    with pytest.raises(ValueError, match="audit selection commitment mismatch"):
        verify_keyed_replay(decoded, {3: KEY})


def test_v33_retention_floor_is_recomputed_locally_and_manifest_must_match():
    artifact = encode_compact_sequence(rows(9, 12), wire_version=VERSION_V33, audit_key=KEY, audit_key_id=3, stochastic_rate_ppm=1_000_000, spans=[{"start_token": 0, "end_token": 2, "signal_id": 7, "score": 0.5, "provenance_id": 4, "evaluator_id": 99}])
    decoded = decode_compact_sequence(artifact)
    summary = compute_evidence_budget_summary(artifact, "DSA-CI-Lite")
    assert summary.minimum_reconstructable_bytes == decoded["manifest"].minimum_reconstructable_bytes == len(artifact)
    with pytest.raises(ValueError, match="floor mismatch"):
        compute_evidence_budget_summary(_rewrite_manifest_floor(artifact, summary.minimum_reconstructable_bytes - 1), "DSA-CI-Lite")
    with pytest.raises(ValueError, match="floor mismatch"):
        compute_evidence_budget_summary(_rewrite_manifest_floor(artifact, summary.minimum_reconstructable_bytes + 1), "DSA-CI-Lite")
