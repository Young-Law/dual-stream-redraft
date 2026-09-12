from __future__ import annotations

import binascii
import hashlib
import hmac
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .compact_evidence import (
    ADAPTIVE_POLICY_FIXED,
    ADAPTIVE_POLICY_FIXED_ID,
    ADAPTIVE_POLICY_HYBRID,
    ADAPTIVE_POLICY_HYBRID_ID,
    MAGIC,
    PREFIX,
    RECORD_HAS_FALLBACK_CHOSEN_ID,
    SPAN_HAS_EVALUATOR_ID,
    TRIGGER_CANARY,
    TRIGGER_HISTORY,
    TRIGGER_RANK,
    TRIGGER_STOCHASTIC,
    VERSION_V33,
    ZERO_HASH,
    EvidenceManifestV33,
    MonologueSequenceHeaderV3,
    _CHUNK_V33,
    _HEADER_V33,
    _MANIFEST_V33,
    _SPAN_V33,
    _SPAN_V33_EVAL,
    _TOKEN_V33_PREFIX,
    _TOPK_V33,
    _manifest_from_fields,
    _profile_from_enum,
    audit_selection_commitment,
    keyed_sample_selected,
)
from .evidence_profile import V33_HEADER_FIELD_RANGES, get_evidence_profile


@dataclass(frozen=True)
class StreamingWorkCounters:
    bytes_read: int
    bytes_hashed: int
    token_records_decoded: int
    candidate_entries_decoded: int
    varint_bytes_decoded: int
    chunks_verified: int
    span_events_indexed: int
    span_overlay_operations: int
    allocations: int
    maximum_live_bytes: int
    full_artifact_materializations: int = 0


@dataclass(frozen=True)
class StreamingVerificationResult:
    header: MonologueSequenceHeaderV3
    manifest: EvidenceManifestV33
    metadata: dict[str, object]
    artifact_sha256: str
    raw_bytes: int
    candidate_entries: int
    adaptive_record_count: int
    max_effective_topk: int
    rank_overflow_count: int
    counters: StreamingWorkCounters


class _Reader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.bytes_read = 0
        self.artifact_hash = hashlib.sha256()
        self.content_hash = hashlib.sha256()
        self.bytes_hashed = 0
        self.before_manifest = True

    def read_exact(self, size: int, label: str, *, hash_artifact: bool = True) -> bytes:
        data = self.stream.read(size)
        self.bytes_read += len(data)
        if len(data) != size:
            raise ValueError(f"artifact is truncated in {label}")
        if hash_artifact:
            self.artifact_hash.update(data)
            self.bytes_hashed += len(data)
            if self.before_manifest:
                self.content_hash.update(data)
                self.bytes_hashed += len(data)
        return data


def _validate_field(name: str, value: int) -> None:
    low, high = V33_HEADER_FIELD_RANGES.get(name, (0, 0xFFFFFFFF))
    if not low <= value <= high:
        raise ValueError(
            f"V3.3 header field {name} value {value} is outside valid range [{low}, {high}]"
        )


def _merkle_root_from_hashes(hashes: list[bytes]) -> tuple[bytes, int]:
    if not hashes:
        return hashlib.sha256(b"").digest(), 0
    level = list(hashes)
    hashed_bytes = 0
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        next_level = []
        for index in range(0, len(level), 2):
            value = level[index] + level[index + 1]
            next_level.append(hashlib.sha256(value).digest())
            hashed_bytes += len(value)
        level = next_level
    return level[0], hashed_bytes


def _decode_payload(
    payload: bytes,
    *,
    start: int,
    count: int,
    base_topk: int,
    max_adaptive_rank: int,
    commit_hash: hashlib._Hash,
    eligibility_hash: hashlib._Hash,
    adaptive_k: bool,
) -> tuple[int, int, int, int, dict[str, int], dict[int, int], int]:
    pos = 0
    candidates = adaptive_count = rank_overflow = max_effective = 0
    triggers = {"rank": 0, "stochastic": 0, "history": 0, "canary": 0}
    histogram = {3: 0, 5: 0, 10: 0, 255: 0}
    hashed_bytes = 0
    for offset in range(count):
        if pos + _TOKEN_V33_PREFIX.size > len(payload):
            raise ValueError("malformed V3.3 token evidence")
        delta, chosen_rank_raw, trigger_flags, record_flags = _TOKEN_V33_PREFIX.unpack_from(payload, pos)
        pos += _TOKEN_V33_PREFIX.size
        if record_flags & ~RECORD_HAS_FALLBACK_CHOSEN_ID:
            raise ValueError("unknown V3.3 token record flags")
        effective_topk = base_topk + delta
        if effective_topk > max_adaptive_rank:
            raise ValueError("V3.3 token evidence exceeds declared maximum adaptive K")
        chosen_id = None
        if record_flags & RECORD_HAS_FALLBACK_CHOSEN_ID:
            if pos + 4 > len(payload):
                raise ValueError("malformed V3.3 fallback chosen-id evidence")
            chosen_id = struct.unpack_from("<I", payload, pos)[0]
            pos += 4
        ids: list[int] = []
        base_candidate_bytes = bytearray()
        for candidate_index in range(effective_topk):
            if pos + _TOPK_V33.size > len(payload):
                raise ValueError("malformed V3.3 top-k evidence")
            candidate = payload[pos : pos + _TOPK_V33.size]
            token_id, _score = _TOPK_V33.unpack(candidate)
            pos += _TOPK_V33.size
            ids.append(token_id)
            if candidate_index < base_topk:
                base_candidate_bytes.extend(candidate)
        if chosen_id is None:
            chosen_rank = int(chosen_rank_raw)
            if chosen_rank < 1 or chosen_rank > len(ids):
                raise ValueError("V3.3 chosen rank is outside retained candidates")
            chosen_id = ids[chosen_rank - 1]
        else:
            chosen_rank = ids.index(chosen_id) + 1 if chosen_id in ids else 255

        token_index = start + offset
        commit_input = struct.pack("<II", token_index, chosen_id) + base_candidate_bytes
        commit_hash.update(commit_input)
        hashed_bytes += len(commit_input)
        raw_rank = chosen_rank if chosen_rank != 255 else max_adaptive_rank + 1
        rank_eligible = adaptive_k and base_topk < raw_rank <= max_adaptive_rank
        eligibility_input = struct.pack(
            "<IBBB",
            token_index,
            int(rank_eligible),
            int(bool(trigger_flags & TRIGGER_HISTORY)),
            int(bool(trigger_flags & TRIGGER_CANARY)),
        )
        eligibility_hash.update(eligibility_input)
        hashed_bytes += len(eligibility_input)

        if bool(trigger_flags & TRIGGER_RANK) != rank_eligible:
            raise ValueError("pre-stochastic rank eligibility mismatch")
        candidates += effective_topk
        adaptive_count += int(effective_topk > base_topk)
        rank_overflow += int(chosen_rank == 255 or chosen_rank > max_adaptive_rank)
        max_effective = max(max_effective, effective_topk)
        triggers["rank"] += int(bool(trigger_flags & TRIGGER_RANK))
        triggers["stochastic"] += int(bool(trigger_flags & TRIGGER_STOCHASTIC))
        triggers["history"] += int(bool(trigger_flags & TRIGGER_HISTORY))
        triggers["canary"] += int(bool(trigger_flags & TRIGGER_CANARY))
        histogram[effective_topk if effective_topk in (3, 5, 10) else 255] += 1
    if pos != len(payload):
        raise ValueError("malformed V3.3 chunk payload")
    return candidates, adaptive_count, rank_overflow, max_effective, triggers, histogram, hashed_bytes


def _require_string(metadata: dict[str, object], name: str, *, hex_digest: bool = False) -> str:
    value = metadata.get(name)
    if not isinstance(value, str):
        raise ValueError(f"authenticated metadata {name} must be a string")
    if hex_digest and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"authenticated metadata {name} must be a canonical SHA-256 hex digest")
    return value


def _require_int(metadata: dict[str, object], name: str, maximum: int) -> int:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"authenticated metadata {name} is out of range")
    return value


def verify_v33_stream(
    path: str | Path,
    *,
    audit_keys: dict[int, bytes] | None = None,
    initial_bytes_read: int = 0,
) -> StreamingVerificationResult:
    """Verify a V3.3 artifact without loading the artifact or token set in full."""

    artifact_size = Path(path).stat().st_size
    with Path(path).open("rb") as stream:
        reader = _Reader(stream)
        reader.bytes_read = initial_bytes_read
        header_bytes = reader.read_exact(_HEADER_V33.size, "V3.3 header")
        fields = _HEADER_V33.unpack(header_bytes)
        (
            magic, version, profile_enum, assurance_enum, prompt_nonce, sequence_id,
            tokenizer_id, signal_schema_id, signal_schema_hash, probe_pack_id,
            probe_pack_hash, decoder_control_flags, base_topk, max_adaptive_rank,
            adaptive_policy_id, stochastic_rate_ppm, audit_key_id,
            audit_selection_commitment_bytes, tension_map_id, tension_map_hash,
            quantization_id, chunk_token_capacity, verifier_work_profile_id,
            runtime_calibration_id, retention_policy_id, metadata_len, chunk_count,
            span_count, token_count,
        ) = fields
        if magic != MAGIC or version != VERSION_V33:
            raise ValueError("compact evidence schema mismatch")
        profile_id = _profile_from_enum(profile_enum)
        profile = get_evidence_profile(profile_id)
        if assurance_enum != 0:
            raise ValueError("software-only decoder supports DSA-R assurance class only")
        if base_topk != profile.base_k or not 1 <= base_topk <= max_adaptive_rank <= 255:
            raise ValueError("V3.3 profile/top-k declaration mismatch")
        if not 0 <= stochastic_rate_ppm <= 1_000_000:
            raise ValueError("invalid V3.3 stochastic sampling rate")
        if chunk_token_capacity < 1:
            raise ValueError("invalid V3.3 chunk token capacity")
        for name, value in (
            ("tokenizer_id", tokenizer_id), ("signal_schema_id", signal_schema_id),
            ("quantization_id", quantization_id),
            ("verifier_work_profile_id", verifier_work_profile_id),
            ("runtime_calibration_id", runtime_calibration_id),
            ("retention_policy_id", retention_policy_id),
        ):
            _validate_field(name, value)
        expected_fields = profile.v33_header_fields
        if (
            tokenizer_id != expected_fields.tokenizer_id
            or signal_schema_id != expected_fields.signal_schema_id
            or quantization_id != expected_fields.quantization_id
            or verifier_work_profile_id != expected_fields.verifier_work_profile_id
            or runtime_calibration_id != expected_fields.runtime_calibration_id
            or retention_policy_id != expected_fields.retention_policy_id
        ):
            raise ValueError("V3.3 registered profile field mismatch")

        header_digest = hashlib.sha256(header_bytes).hexdigest()
        reader.bytes_hashed += len(header_bytes)
        metadata_digest = reader.read_exact(32, "V3.3 metadata digest")
        metadata_bytes = reader.read_exact(metadata_len, "V3.3 metadata")
        reader.bytes_hashed += len(metadata_bytes)
        if hashlib.sha256(metadata_bytes).digest() != metadata_digest:
            raise ValueError("V3.3 metadata digest mismatch")
        try:
            metadata = json.loads(metadata_bytes.decode("utf-8")) if metadata_bytes else {}
        except Exception as exc:
            raise ValueError("malformed V3.3 metadata") from exc
        if not isinstance(metadata, dict):
            raise ValueError("V3.3 metadata must be an object")
        if metadata.get("profile_id") != profile_id:
            raise ValueError("V3.3 metadata profile declaration mismatch")
        del metadata_bytes

        adaptive_k = adaptive_policy_id == ADAPTIVE_POLICY_HYBRID_ID
        if adaptive_policy_id not in (ADAPTIVE_POLICY_FIXED_ID, ADAPTIVE_POLICY_HYBRID_ID):
            raise ValueError("V3.3 adaptive policy semantics mismatch")
        if adaptive_k and max_adaptive_rank <= base_topk:
            raise ValueError("V3.3 adaptive policy/header semantics mismatch")
        if not adaptive_k and max_adaptive_rank != base_topk:
            raise ValueError("V3.3 fixed policy/header semantics mismatch")

        commit_hash = hashlib.sha256()
        eligibility_hash = hashlib.sha256()
        chunk_hashes: list[bytes] = []
        chunks_start = stream.tell()
        expected_start = 0
        candidates = adaptive_count = rank_overflow = max_effective = 0
        total_triggers = {"rank": 0, "stochastic": 0, "history": 0, "canary": 0}
        total_histogram = {3: 0, 5: 0, 10: 0, 255: 0}
        max_payload_len = 0
        allocations = 2
        for expected_chunk in range(chunk_count):
            chunk_header = reader.read_exact(_CHUNK_V33.size, "V3.3 chunk header")
            (
                chunk_index, first_token_index, chunk_token_count, chunk_flags,
                chunk_base_topk, declared_max_topk, rank_count, stochastic_count,
                history_count, canary_count, payload_len, chunk_crc32, chunk_hash,
            ) = _CHUNK_V33.unpack(chunk_header)
            if chunk_index != expected_chunk or first_token_index != expected_start:
                raise ValueError("compact chunks are missing or reordered")
            expected_chunk_flags = 1 if expected_chunk + 1 == chunk_count else 0
            if chunk_flags != expected_chunk_flags:
                raise ValueError("unknown V3.3 chunk flags")
            if chunk_base_topk != base_topk or chunk_token_count > chunk_token_capacity:
                raise ValueError("V3.3 chunk declaration mismatch")
            maximum_payload = chunk_token_count * (_TOKEN_V33_PREFIX.size + 4 + _TOPK_V33.size * max_adaptive_rank)
            if payload_len > maximum_payload:
                raise ValueError("V3.3 chunk payload exceeds declared decoding bound")
            payload = reader.read_exact(payload_len, "V3.3 chunk payload")
            max_payload_len = max(max_payload_len, payload_len)
            reader.bytes_hashed += len(payload)
            actual_chunk_hash = hashlib.sha256(payload).digest()
            if binascii.crc32(payload) & 0xFFFFFFFF != chunk_crc32 or actual_chunk_hash != chunk_hash:
                raise ValueError("compact chunk integrity check failed")
            chunk_hashes.append(actual_chunk_hash)
            allocations += 3 + chunk_token_count
            (chunk_candidates, chunk_adaptive, chunk_overflow, chunk_max,
             chunk_triggers, chunk_histogram, token_hashed_bytes) = _decode_payload(
                payload, start=first_token_index, count=chunk_token_count,
                base_topk=base_topk, max_adaptive_rank=max_adaptive_rank,
                commit_hash=commit_hash, eligibility_hash=eligibility_hash,
                adaptive_k=adaptive_k,
            )
            reader.bytes_hashed += token_hashed_bytes
            if chunk_max != declared_max_topk:
                raise ValueError("V3.3 maximum effective top-k mismatch")
            declared_triggers = (rank_count, stochastic_count, history_count, canary_count)
            observed_triggers = tuple(chunk_triggers[name] for name in ("rank", "stochastic", "history", "canary"))
            if declared_triggers != observed_triggers:
                raise ValueError("V3.3 chunk trigger count mismatch")
            candidates += chunk_candidates
            adaptive_count += chunk_adaptive
            rank_overflow += chunk_overflow
            max_effective = max(max_effective, chunk_max)
            for name in total_triggers:
                total_triggers[name] += chunk_triggers[name]
            for key in total_histogram:
                total_histogram[key] += chunk_histogram[key]
            expected_start += chunk_token_count
            del payload
        if expected_start != token_count:
            raise ValueError("missing token evidence")

        span_bytes = 0
        for _ in range(span_count):
            span = reader.read_exact(_SPAN_V33.size, "V3.3 span events")
            start, end, signal_id, q_score, provenance_id = _SPAN_V33.unpack(span)
            flags = reader.read_exact(1, "V3.3 span flags")[0]
            span_bytes += _SPAN_V33.size + 1
            if flags & SPAN_HAS_EVALUATOR_ID:
                reader.read_exact(_SPAN_V33_EVAL.size, "V3.3 span evaluator")
                span_bytes += _SPAN_V33_EVAL.size
            if flags & ~SPAN_HAS_EVALUATOR_ID:
                raise ValueError("unknown V3.3 span flags")
            if start >= end or end > token_count:
                raise ValueError("sparse span is outside token range")
            allocations += 1

        reader.before_manifest = False
        manifest_bytes = reader.read_exact(_MANIFEST_V33.size, "V3.3 manifest")
        trailing = stream.read(1)
        reader.bytes_read += len(trailing)
        if trailing:
            raise ValueError("unexpected trailing V3.3 artifact bytes")
        zeroed_manifest = ZERO_HASH + manifest_bytes[32:]
        reader.content_hash.update(zeroed_manifest)
        reader.bytes_hashed += len(zeroed_manifest)
        manifest = _manifest_from_fields(_MANIFEST_V33.unpack(manifest_bytes))
        allocations += 1
        merkle_root, merkle_hashed = _merkle_root_from_hashes(chunk_hashes)
        reader.bytes_hashed += merkle_hashed
        if manifest.token_count != token_count or manifest.chunk_count != chunk_count or manifest.span_event_count != span_count:
            raise ValueError("V3.3 manifest count mismatch")
        if manifest.sequence_header_hash != header_digest:
            raise ValueError("V3.3 sequence header hash mismatch")
        if manifest.chunk_merkle_root != merkle_root.hex():
            raise ValueError("V3.3 chunk Merkle root mismatch")
        if manifest.artifact_content_hash != reader.content_hash.hexdigest():
            raise ValueError("V3.3 artifact content hash mismatch")
        if manifest.raw_evidence_bytes != artifact_size:
            raise ValueError("V3.3 raw evidence byte count mismatch")
        token_wire_bytes = artifact_size - _HEADER_V33.size - 32 - metadata_len - chunk_count * _CHUNK_V33.size - span_bytes - _MANIFEST_V33.size
        minimum = _HEADER_V33.size + 32 + metadata_len + chunk_count * _CHUNK_V33.size + token_wire_bytes + span_bytes + _MANIFEST_V33.size
        if manifest.minimum_reconstructable_bytes != minimum:
            raise ValueError("V3.3 minimum reconstructable byte floor mismatch")
        if manifest.effective_k_histogram != total_histogram:
            raise ValueError("V3.3 effective-k histogram mismatch")
        for name in ("rank", "stochastic", "history", "canary"):
            if manifest.trigger_count_by_reason[name] != total_triggers[name]:
                raise ValueError("V3.3 manifest trigger count mismatch")

        commit_identity = commit_hash.hexdigest()
        eligibility_digest = eligibility_hash.hexdigest()
        if audit_keys is not None:
            key_id = audit_key_id
            if key_id not in audit_keys:
                raise ValueError("unknown audit key id")
            if not hmac.compare_digest(_require_string(metadata, "commit_identity", hex_digest=True), commit_identity):
                raise ValueError("commit identity mismatch")
            if not hmac.compare_digest(_require_string(metadata, "pre_stochastic_eligibility_digest", hex_digest=True), eligibility_digest):
                raise ValueError("pre-stochastic eligibility mismatch")
            metadata_nonce = _require_int(metadata, "prompt_nonce", 0xFFFFFFFFFFFFFFFF)
            if metadata_nonce != prompt_nonce or sequence_id != (prompt_nonce & 0xFFFFFFFF):
                raise ValueError("V3.3 replay identifier/sequence id mismatch before commitment verification")
            benchmark_id = _require_string(metadata, "benchmark_id")
            policy_name = _require_string(metadata, "adaptive_policy")
            expected_policy = ADAPTIVE_POLICY_HYBRID if adaptive_k else ADAPTIVE_POLICY_FIXED
            if policy_name != expected_policy:
                raise ValueError("V3.3 adaptive policy semantics mismatch")
            if _require_string(metadata, "profile_id") != profile_id:
                raise ValueError("V3.3 authenticated profile mismatch")
            canary_eval = metadata.get("canary_eval")
            if type(canary_eval) is not bool:
                raise ValueError("authenticated metadata canary_eval must be a boolean")
            if not canary_eval and total_triggers["canary"]:
                raise ValueError("pre-stochastic eligibility has canary evidence without an authenticated evaluation label")
            rate_ppm = stochastic_rate_ppm
            policy_version = adaptive_policy_id
            commitment = audit_selection_commitment(
                audit_keys[key_id], commit_identity=commit_identity, sequence_id=prompt_nonce,
                policy_version=policy_version, rate_ppm=rate_ppm, benchmark_id=benchmark_id,
                eligibility_digest=eligibility_digest, audit_key_id=key_id,
                profile_id=profile_id, base_k=base_topk,
                max_adaptive_k=max_adaptive_rank, adaptive_policy=policy_name,
                canary_eval=canary_eval,
            ).hex()
            serialized_commitment = audit_selection_commitment_bytes.hex()
            if not hmac.compare_digest(commitment, serialized_commitment):
                raise ValueError("audit selection commitment mismatch")
            # A second sequential pass checks stochastic decisions after the
            # commit identity is known, preserving O(chunk-size) live memory.
            stream.seek(chunks_start)
            for _ in range(chunk_count):
                chunk_header = reader.read_exact(_CHUNK_V33.size, "V3.3 replay chunk header", hash_artifact=False)
                chunk_fields = _CHUNK_V33.unpack(chunk_header)
                first_token_index, chunk_token_count, payload_len = chunk_fields[1], chunk_fields[2], chunk_fields[10]
                payload = reader.read_exact(payload_len, "V3.3 replay chunk payload", hash_artifact=False)
                pos = 0
                for offset in range(chunk_token_count):
                    delta, chosen_rank, flags, record_flags = _TOKEN_V33_PREFIX.unpack_from(payload, pos)
                    pos += _TOKEN_V33_PREFIX.size
                    fallback_chosen_id = None
                    if record_flags & RECORD_HAS_FALLBACK_CHOSEN_ID:
                        fallback_chosen_id = struct.unpack_from("<I", payload, pos)[0]
                        pos += 4
                    effective_topk = base_topk + delta
                    fallback_rank = 255
                    for candidate_rank in range(1, effective_topk + 1):
                        candidate_id, _score = _TOPK_V33.unpack_from(payload, pos)
                        pos += _TOPK_V33.size
                        if fallback_chosen_id is not None and candidate_id == fallback_chosen_id:
                            fallback_rank = candidate_rank
                    raw_rank = fallback_rank if fallback_chosen_id is not None else chosen_rank
                    if raw_rank == 255:
                        raw_rank = max_adaptive_rank + 1
                    rank_eligible = adaptive_k and base_topk < raw_rank <= max_adaptive_rank
                    otherwise_untriggered = not (rank_eligible or flags & TRIGGER_HISTORY or flags & TRIGGER_CANARY)
                    expected = otherwise_untriggered and bool(rate_ppm) and keyed_sample_selected(
                        audit_keys[key_id], commit_identity=commit_identity,
                        sequence_id=prompt_nonce, token_index=first_token_index + offset,
                        policy_version=policy_version, rate_ppm=rate_ppm,
                        benchmark_id=benchmark_id, audit_key_id=key_id,
                        profile_id=profile_id, base_k=base_topk,
                        max_adaptive_k=max_adaptive_rank, adaptive_policy=policy_name,
                        canary_eval=canary_eval,
                    )
                    if bool(flags & TRIGGER_STOCHASTIC) != expected:
                        raise ValueError("keyed stochastic selection replay mismatch")
                del payload

        maximum_live_bytes = max(
            _HEADER_V33.size + 32 + metadata_len,
            _CHUNK_V33.size + max_payload_len + 32 * chunk_count,
            _MANIFEST_V33.size + 32 * chunk_count,
        )
        counters = StreamingWorkCounters(
            bytes_read=reader.bytes_read,
            bytes_hashed=reader.bytes_hashed,
            token_records_decoded=token_count * (2 if audit_keys is not None else 1),
            candidate_entries_decoded=candidates * (2 if audit_keys is not None else 1),
            varint_bytes_decoded=0,
            chunks_verified=chunk_count,
            span_events_indexed=span_count,
            span_overlay_operations=0,
            allocations=allocations,
            maximum_live_bytes=maximum_live_bytes,
        )
        return StreamingVerificationResult(
            header=MonologueSequenceHeaderV3(sequence_id, token_count, profile_id, base_topk, max_adaptive_rank, chunk_token_capacity, schema_version=VERSION_V33),
            manifest=manifest,
            metadata={
                **metadata,
                "audit_key_id": audit_key_id,
                "audit_selection_commitment": audit_selection_commitment_bytes.hex(),
                "header_prompt_nonce": prompt_nonce,
                "verifier_work_profile_id": verifier_work_profile_id,
            },
            artifact_sha256=reader.artifact_hash.hexdigest(), raw_bytes=artifact_size,
            candidate_entries=candidates, adaptive_record_count=adaptive_count,
            max_effective_topk=max_effective, rank_overflow_count=rank_overflow,
            counters=counters,
        )
