from __future__ import annotations

import binascii
import hashlib
import hmac
import json
import struct
from dataclasses import dataclass
from typing import Any, Iterable

from .evidence_profile import get_evidence_profile

MAGIC = b"DSAEV29\0"
VERSION_V31 = 0x0301
VERSION_V32 = 0x0302
VERSION_V33 = 0x0303
VERSION = VERSION_V32
SCORE_SCALE = 255.0
SCORE_TOLERANCE = 0.5 / SCORE_SCALE
ZERO_HASH = b"\0" * 32

PREFIX = struct.Struct("<8sH")
_PREFIX = PREFIX
_HEADER_V31 = struct.Struct("<8sHBBQIIHHHH")
_HEADER_V32 = struct.Struct("<8sHBHQIIHHHH")
_HEADER = _HEADER_V32
_CHUNK = struct.Struct("<IIIII")
_TOKEN = struct.Struct("<IBB")
_TOPK = struct.Struct("<IB")
_SPAN = struct.Struct("<IIHB")

# V3.3 binary layout. All integers are little-endian. Variable sections are
# bounded by the fixed counts and byte lengths declared in these records.
_HEADER_V33 = struct.Struct("<8sHBBQIHH32sI32sI B B H I I 32s I 32s B H H H H H H H H")
_CHUNK_V33 = struct.Struct("<IIHHBBHHHHII32s")
_TOKEN_V33_PREFIX = struct.Struct("<BBBB")
_TOPK_V33 = struct.Struct("<IB")
_SPAN_V33 = struct.Struct("<IIHBI")
_SPAN_V33_EVAL = struct.Struct("<I")
_MANIFEST_V33 = struct.Struct("<32s32s32sIII II IIII HHHHHH 32s B 32s")
_TRIGGER_NAMES = ("rank", "stochastic", "history", "canary", "multi", "escalation")

TRIGGER_RANK = 0x01
TRIGGER_STOCHASTIC = 0x02
TRIGGER_HISTORY = 0x04
TRIGGER_CANARY = 0x08
TRIGGER_ESCALATION = 0x10
RECORD_HAS_FALLBACK_CHOSEN_ID = 0x01
SPAN_HAS_EVALUATOR_ID = 0x01
MANIFEST_HASH_OFFSET_FROM_END = _MANIFEST_V33.size


@dataclass(frozen=True)
class MonologueSequenceHeaderV3:
    sequence_id: int
    token_count: int
    profile_id: str
    base_k: int
    max_adaptive_k: int
    chunk_token_capacity: int
    score_tolerance: float = SCORE_TOLERANCE
    schema_version: int = VERSION


@dataclass(frozen=True)
class EvidenceChunkV3:
    chunk_index: int
    start_token: int
    token_count: int
    crc32: int = 0


@dataclass(frozen=True)
class CompactTokenEvidenceV3:
    token_index: int
    chosen_id: int
    topk_ids: tuple[int, ...]
    topk_scores: tuple[float, ...]
    effective_topk: int
    chosen_rank: int = 255
    trigger_flags: int = 0
    record_flags: int = 0


@dataclass(frozen=True)
class SignalSpanEventV3:
    start_token: int
    end_token: int
    signal_id: int
    score: float
    provenance_id: int = 0
    evaluator_id: int | None = None


@dataclass(frozen=True)
class EvidenceManifestV33:
    artifact_content_hash: str
    sequence_header_hash: str
    chunk_merkle_root: str
    token_count: int
    chunk_count: int
    span_event_count: int
    raw_evidence_bytes: int
    minimum_reconstructable_bytes: int
    effective_k_histogram: dict[int, int]
    trigger_count_by_reason: dict[str, int]
    audit_selection_digest: str
    local_audit_outcome: str = "LOCAL_PASS"
    retention_requirement_hash: str = "0" * 64


def quantize_score(score: float) -> int:
    return max(0, min(255, int(round(float(score) * SCORE_SCALE))))


def dequantize_score(raw: int) -> float:
    return int(raw) / SCORE_SCALE


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _int_hash(text: str | None, bits: int) -> int:
    if not text:
        return 0
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    return value & ((1 << bits) - 1)


def _metadata_hash(metadata: dict[str, Any]) -> bytes:
    return _sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _merkle_root(chunks: list[bytes]) -> bytes:
    if not chunks:
        return _sha256(b"")
    level = [_sha256(chunk) for chunk in chunks]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_sha256(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def choose_effective_topk(
    chosen_id: int,
    candidate_ids: list[int] | tuple[int, ...],
    base_k: int,
    max_adaptive_k: int,
    adaptive: bool = True,
) -> int:
    base = min(int(base_k), len(candidate_ids))
    if not adaptive:
        return base
    try:
        rank = list(candidate_ids).index(int(chosen_id)) + 1
    except ValueError:
        rank = len(candidate_ids) + 1
    if rank <= base:
        return base
    return min(max(int(max_adaptive_k), base), max(rank, base))


def _normalise_tokens(
    tokens: Iterable[Any],
    base_k: int,
    max_adaptive_k: int,
    adaptive: bool,
) -> list[CompactTokenEvidenceV3]:
    out: list[CompactTokenEvidenceV3] = []
    for idx, rec in enumerate(tokens):
        if isinstance(rec, CompactTokenEvidenceV3):
            out.append(rec)
            continue
        is_dict = isinstance(rec, dict)
        token_index = int(rec.get("token_index", idx) if is_dict else getattr(rec, "token_index", idx))
        chosen_id = int(rec["chosen_id"] if is_dict else getattr(rec, "chosen_id"))
        topk = rec.get("topk") if is_dict else getattr(rec, "topk", None)
        if topk is not None:
            ids = [int(t["token_id"] if isinstance(t, dict) else getattr(t, "token_id")) for t in topk]
            scores = [float(t["prob"] if isinstance(t, dict) else getattr(t, "prob")) for t in topk]
        else:
            ids = [int(x) for x in (rec["topk_ids"] if is_dict else getattr(rec, "topk_ids"))]
            scores = [float(x) for x in (rec["topk_scores"] if is_dict else getattr(rec, "topk_scores"))]
        requested = rec.get("effective_topk") if is_dict else getattr(rec, "effective_topk", 0)
        eff = int(requested or choose_effective_topk(chosen_id, ids, base_k, max_adaptive_k, adaptive))
        eff = min(eff, len(ids), max_adaptive_k)
        kept_ids = ids[:eff]
        rank = kept_ids.index(chosen_id) + 1 if chosen_id in kept_ids else 255
        flags = int(rec.get("trigger_flags", 0) if is_dict else getattr(rec, "trigger_flags", 0))
        record_flags = int(rec.get("record_flags", 0) if is_dict else getattr(rec, "record_flags", 0))
        out.append(
            CompactTokenEvidenceV3(
                token_index=token_index,
                chosen_id=chosen_id,
                topk_ids=tuple(kept_ids),
                topk_scores=tuple(scores[:eff]),
                effective_topk=eff,
                chosen_rank=rank,
                trigger_flags=flags,
                record_flags=record_flags,
            )
        )
    return out


def keyed_sample_selected(
    key: bytes,
    *,
    commit_identity: str,
    sequence_id: int,
    token_index: int,
    policy_version: int,
    rate_ppm: int,
    benchmark_id: str = "",
    domain: str = "DSA-v2.10-keyed-sample",
) -> bool:
    context = (
        f"{domain}\0{commit_identity}\0{sequence_id}\0{token_index}"
        f"\0{policy_version}\0{benchmark_id}"
    ).encode("utf-8")
    value = int.from_bytes(hmac.new(key, context, hashlib.sha256).digest()[:8], "big")
    return value % 1_000_000 < int(rate_ppm)


def audit_selection_commitment(
    key: bytes,
    *,
    commit_identity: str,
    sequence_id: int,
    policy_version: int,
    rate_ppm: int,
    benchmark_id: str = "",
    eligibility_digest: str = "",
    domain: str = "DSA-v2.10-keyed-sample",
) -> bytes:
    public = (
        f"{domain}\0{commit_identity}\0{sequence_id}\0{policy_version}"
        f"\0{rate_ppm}\0{benchmark_id}\0{eligibility_digest}"
    ).encode("utf-8")
    return hmac.new(key, public, hashlib.sha256).digest()


def _commit_identity(records: list[CompactTokenEvidenceV3], base_k: int) -> str:
    h = hashlib.sha256()
    for rec in records:
        h.update(struct.pack("<II", rec.token_index, rec.chosen_id))
        for token_id, score in zip(rec.topk_ids[:base_k], rec.topk_scores[:base_k]):
            h.update(_TOPK_V33.pack(int(token_id), quantize_score(score)))
    return h.hexdigest()


def _pre_stochastic_eligibility_digest(records: list[CompactTokenEvidenceV3]) -> str:
    """Bind the exact Phase-1 rank/history/canary state evaluated at encode time.

    History and canary provenance is deliberately not independently established by
    the Phase-1 verifier, but this digest prevents those states from being changed
    after keyed selection has been committed.
    """
    state = bytearray()
    mask = TRIGGER_RANK | TRIGGER_HISTORY | TRIGGER_CANARY
    for rec in records:
        state += struct.pack("<IB", rec.token_index, rec.trigger_flags & mask)
    return _sha256_hex(bytes(state))


def _apply_v33_triggers(
    records: list[CompactTokenEvidenceV3],
    *,
    base_k: int,
    max_adaptive_rank: int,
    adaptive_k: bool,
    audit_key: bytes | None,
    audit_key_id: int,
    sequence_id: int,
    stochastic_rate_ppm: int,
    policy_version: int,
    benchmark_id: str,
    canary_eval: bool,
) -> tuple[list[CompactTokenEvidenceV3], bytes, str, str]:
    commit = _commit_identity(records, base_k)
    if stochastic_rate_ppm and audit_key is None:
        raise ValueError("keyed stochastic sampling requires an audit key")

    out: list[CompactTokenEvidenceV3] = []
    for rec in records:
        flags = rec.trigger_flags
        effective_topk = base_k
        raw_rank = rec.chosen_rank if rec.chosen_rank != 255 else max_adaptive_rank + 1
        if adaptive_k and raw_rank > base_k and raw_rank <= max_adaptive_rank:
            flags |= TRIGGER_RANK
            effective_topk = max(effective_topk, raw_rank)
        if flags & TRIGGER_CANARY and not canary_eval:
            raise ValueError("canary evidence is only allowed in explicitly labeled evaluation runs")
        # Stochastic selection is evaluated only after this canonical state is
        # finalized. Replay authenticates this exact state before trusting an
        # exemption.
        otherwise_untriggered = not (flags & (TRIGGER_RANK | TRIGGER_HISTORY | TRIGGER_CANARY))
        if otherwise_untriggered and audit_key is not None and stochastic_rate_ppm:
            if keyed_sample_selected(
                audit_key,
                commit_identity=commit,
                sequence_id=sequence_id,
                token_index=rec.token_index,
                policy_version=policy_version,
                rate_ppm=stochastic_rate_ppm,
                benchmark_id=benchmark_id,
            ):
                flags |= TRIGGER_STOCHASTIC
                effective_topk = max(effective_topk, min(max_adaptive_rank, len(rec.topk_ids)))
        effective_topk = min(effective_topk, len(rec.topk_ids), max_adaptive_rank)
        kept_ids = rec.topk_ids[:effective_topk]
        kept_scores = rec.topk_scores[:effective_topk]
        chosen_rank = kept_ids.index(rec.chosen_id) + 1 if rec.chosen_id in kept_ids else 255
        out.append(
            CompactTokenEvidenceV3(
                token_index=rec.token_index,
                chosen_id=rec.chosen_id,
                topk_ids=tuple(kept_ids),
                topk_scores=tuple(kept_scores),
                effective_topk=effective_topk,
                chosen_rank=chosen_rank,
                trigger_flags=flags,
                record_flags=rec.record_flags,
            )
        )
    eligibility_digest = _pre_stochastic_eligibility_digest(out)
    commitment = ZERO_HASH
    if audit_key is not None:
        commitment = audit_selection_commitment(
            audit_key,
            commit_identity=commit,
            sequence_id=sequence_id,
            policy_version=policy_version,
            rate_ppm=stochastic_rate_ppm,
            benchmark_id=benchmark_id,
            eligibility_digest=eligibility_digest,
        )
    return out, commitment, commit, eligibility_digest


def encode_compact_sequence(
    tokens: Iterable[Any],
    *,
    profile: str = "DSA-CI-Lite",
    sequence_id: int = 0,
    chunk_token_capacity: int = 256,
    adaptive_k: bool = True,
    max_adaptive_k: int | None = None,
    spans: Iterable[SignalSpanEventV3 | dict[str, Any]] = (),
    wire_version: int = VERSION_V32,
    audit_key: bytes | None = None,
    audit_key_id: int = 0,
    stochastic_rate_ppm: int = 0,
    benchmark_id: str = "",
    canary_eval: bool = False,
    assurance_class: str = "DSA-R",
) -> bytes:
    if wire_version == VERSION_V33:
        return encode_compact_sequence_v33(
            tokens,
            profile=profile,
            sequence_id=sequence_id,
            chunk_token_capacity=chunk_token_capacity,
            adaptive_k=adaptive_k,
            max_adaptive_k=max_adaptive_k,
            spans=spans,
            audit_key=audit_key,
            audit_key_id=audit_key_id,
            stochastic_rate_ppm=stochastic_rate_ppm,
            benchmark_id=benchmark_id,
            canary_eval=canary_eval,
            assurance_class=assurance_class,
        )
    if wire_version != VERSION_V32:
        raise ValueError(f"unsupported compact evidence encode version 0x{wire_version:04x}")
    return _encode_compact_sequence_v32(
        tokens,
        profile=profile,
        sequence_id=sequence_id,
        chunk_token_capacity=chunk_token_capacity,
        adaptive_k=adaptive_k,
        max_adaptive_k=max_adaptive_k,
        spans=spans,
    )


def _encode_compact_sequence_v32(
    tokens: Iterable[Any],
    *,
    profile: str,
    sequence_id: int,
    chunk_token_capacity: int,
    adaptive_k: bool,
    max_adaptive_k: int | None,
    spans: Iterable[SignalSpanEventV3 | dict[str, Any]],
) -> bytes:
    prof = get_evidence_profile(profile)
    max_k = prof.max_adaptive_k if max_adaptive_k is None else int(max_adaptive_k)
    records = _normalise_tokens(tokens, prof.base_k, max_k, adaptive_k)
    spans_norm = _normalise_spans(spans)
    meta_obj = {
        "evidence_profile": prof.profile_id.value,
        "profile_id": prof.profile_id.value,
        "assurance_class": "DSA-R",
        "signal_schema_id": "dsa-r-v2.9-signals",
        "signal_schema_hash": hashlib.sha256(b"dsa-r-v2.9-signals").hexdigest(),
        "probe_pack_id": "none",
        "probe_pack_hash": hashlib.sha256(b"none").hexdigest(),
        "decoder_control_flags": [],
        "adaptive_policy_id": "rank-triggered-base-k-to-profile-max" if adaptive_k else "fixed-base-k",
        "verifier_budget_id": prof.verifier_budget_id,
        "retention_floor_policy_id": "v2.9-local-reconstructable-floor",
        "quantization_id": "uint8-probability-v1",
    }
    meta = json.dumps(meta_obj, sort_keys=True, separators=(",", ":")).encode()
    chunks = _encode_v32_chunks(records, int(chunk_token_capacity))
    span_body = _encode_v32_spans(spans_norm)
    header = _HEADER.pack(
        MAGIC,
        VERSION_V32,
        len(prof.profile_id.value),
        len(meta),
        int(sequence_id) & 0xFFFFFFFFFFFFFFFF,
        len(records),
        int(chunk_token_capacity),
        prof.base_k,
        max_k,
        len(chunks),
        len(spans_norm),
    )
    return bytes(header + prof.profile_id.value.encode() + meta + b"".join(chunks) + span_body)


def _normalise_spans(spans: Iterable[SignalSpanEventV3 | dict[str, Any]]) -> list[SignalSpanEventV3]:
    out = []
    for span in spans:
        if isinstance(span, SignalSpanEventV3):
            item = span
        else:
            item = SignalSpanEventV3(
                int(span["start_token"]),
                int(span["end_token"]),
                int(span.get("signal_id", span.get("ast_code", 0))),
                float(span.get("score", 0.0)),
                int(span.get("provenance_id", 0)),
                span.get("evaluator_id"),
            )
        if item.start_token >= item.end_token:
            raise ValueError("sparse spans must be non-empty token ranges")
        out.append(item)
    return out


def _encode_v32_chunks(records: list[CompactTokenEvidenceV3], cap: int) -> list[bytes]:
    if cap < 1 or cap > 256:
        raise ValueError("chunk_token_capacity must be between 1 and 256 for compact token offsets")
    chunks: list[bytes] = []
    for chunk_index, start in enumerate(range(0, len(records), cap)):
        subset = records[start:start + cap]
        body = bytearray()
        for local_offset, rec in enumerate(subset):
            expected_token_index = start + local_offset
            if rec.token_index != expected_token_index:
                raise ValueError(f"token evidence index {rec.token_index} does not match expected {expected_token_index}")
            rank = rec.chosen_rank if 1 <= int(rec.chosen_rank) <= len(rec.topk_ids) else 255
            body += _TOKEN.pack(rank, local_offset, rec.effective_topk)
            if rank == 255:
                body += struct.pack("<I", rec.chosen_id)
            for token_id, score in zip(rec.topk_ids, rec.topk_scores):
                body += _TOPK.pack(int(token_id), quantize_score(score))
        crc = binascii.crc32(body) & 0xFFFFFFFF
        chunks.append(_CHUNK.pack(chunk_index, start, len(subset), len(body), crc) + body)
    return chunks


def _encode_v32_spans(spans: list[SignalSpanEventV3]) -> bytes:
    body = bytearray()
    for span in spans:
        body += _SPAN.pack(span.start_token, span.end_token, span.signal_id, quantize_score(span.score))
    return bytes(body)



def _normalise_v33_source_tokens(tokens: Iterable[Any], max_rank: int) -> list[CompactTokenEvidenceV3]:
    records: list[CompactTokenEvidenceV3] = []
    for index, item in enumerate(tokens):
        is_dict = isinstance(item, dict)
        token_index = int(item.get("token_index", index) if is_dict else getattr(item, "token_index", index))
        chosen_id = int(item["chosen_id"] if is_dict else getattr(item, "chosen_id"))
        topk = item.get("topk") if is_dict else getattr(item, "topk", None)
        if topk is not None:
            ids = [int(t["token_id"] if isinstance(t, dict) else getattr(t, "token_id")) for t in topk]
            scores = [float(t["prob"] if isinstance(t, dict) else getattr(t, "prob")) for t in topk]
        else:
            ids = [int(x) for x in (item["topk_ids"] if is_dict else getattr(item, "topk_ids"))]
            scores = [float(x) for x in (item["topk_scores"] if is_dict else getattr(item, "topk_scores"))]
        kept_ids = tuple(ids[:max_rank])
        kept_scores = tuple(scores[:max_rank])
        chosen_rank = kept_ids.index(chosen_id) + 1 if chosen_id in kept_ids else 255
        trigger_flags = int(item.get("trigger_flags", 0) if is_dict else getattr(item, "trigger_flags", 0))
        record_flags = int(item.get("record_flags", 0) if is_dict else getattr(item, "record_flags", 0))
        records.append(CompactTokenEvidenceV3(token_index, chosen_id, kept_ids, kept_scores, len(kept_ids), chosen_rank, trigger_flags, record_flags))
    return records

def encode_compact_sequence_v33(
    tokens: Iterable[Any],
    *,
    profile: str = "DSA-CI-Lite",
    sequence_id: int = 0,
    chunk_token_capacity: int = 256,
    adaptive_k: bool = True,
    max_adaptive_k: int | None = None,
    spans: Iterable[SignalSpanEventV3 | dict[str, Any]] = (),
    audit_key: bytes | None = None,
    audit_key_id: int = 0,
    stochastic_rate_ppm: int = 0,
    benchmark_id: str = "",
    canary_eval: bool = False,
    assurance_class: str = "DSA-R",
) -> bytes:
    prof = get_evidence_profile(profile)
    if assurance_class != "DSA-R":
        raise ValueError("software-only V3.3 generation supports DSA-R only")
    max_rank = prof.max_adaptive_k if max_adaptive_k is None else int(max_adaptive_k)
    base_records = _normalise_v33_source_tokens(tokens, max_rank)
    records, commitment, commit_identity, eligibility_digest = _apply_v33_triggers(
        base_records,
        base_k=prof.base_k,
        max_adaptive_rank=max_rank,
        adaptive_k=adaptive_k,
        audit_key=audit_key,
        audit_key_id=audit_key_id,
        sequence_id=int(sequence_id),
        stochastic_rate_ppm=int(stochastic_rate_ppm),
        policy_version=210,
        benchmark_id=benchmark_id,
        canary_eval=canary_eval,
    )
    spans_norm = _normalise_spans(spans)
    metadata = {
        "profile_id": prof.profile_id.value,
        "adaptive_policy": "dsa-v2.10-hybrid-phase1",
        "commit_identity": commit_identity,
        "pre_stochastic_eligibility_digest": eligibility_digest,
        "prompt_nonce": int(sequence_id) & 0xFFFFFFFFFFFFFFFF,
        "benchmark_id": benchmark_id,
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata_digest = _sha256(metadata_bytes)
    profile_hash = hashlib.sha256(prof.profile_id.value.encode()).digest()
    signal_schema_hash = hashlib.sha256(b"AST-1-v2.10").digest()
    probe_pack_hash = hashlib.sha256(b"none").digest()
    tension_map_hash = hashlib.sha256(b"phase1-none").digest()
    chunks, chunk_payloads = _encode_v33_chunks(records, prof.base_k, int(chunk_token_capacity))
    span_body = _encode_v33_spans(spans_norm, len(records))
    header = _HEADER_V33.pack(
        MAGIC,
        VERSION_V33,
        _profile_enum(profile),
        0,
        int(sequence_id) & 0xFFFFFFFFFFFFFFFF,
        int(sequence_id) & 0xFFFFFFFF,
        1,
        1,
        signal_schema_hash,
        0,
        probe_pack_hash,
        0,
        prof.base_k,
        max_rank,
        210,
        int(stochastic_rate_ppm),
        int(audit_key_id),
        commitment,
        0,
        tension_map_hash,
        1,
        int(chunk_token_capacity),
        1,
        1,
        1,
        len(metadata_bytes),
        len(chunks),
        len(spans_norm),
        len(records),
    )
    pre_manifest = header + metadata_digest + metadata_bytes
    chunk_bytes = b"".join(chunks)
    body_without_manifest = pre_manifest + chunk_bytes + span_body
    minimum_reconstructable = len(body_without_manifest) + _MANIFEST_V33.size
    manifest_zero = _pack_v33_manifest(
        ZERO_HASH,
        _sha256(header),
        _merkle_root(chunk_payloads),
        len(records),
        len(chunks),
        len(spans_norm),
        len(body_without_manifest) + _MANIFEST_V33.size,
        minimum_reconstructable,
        records,
        ZERO_HASH,
    )
    artifact_hash = _sha256(body_without_manifest + manifest_zero)
    manifest = _pack_v33_manifest(
        artifact_hash,
        _sha256(header),
        _merkle_root(chunk_payloads),
        len(records),
        len(chunks),
        len(spans_norm),
        len(body_without_manifest) + _MANIFEST_V33.size,
        minimum_reconstructable,
        records,
        ZERO_HASH,
    )
    return body_without_manifest + manifest


def _profile_enum(profile: str) -> int:
    return {
        "DSA-CI-Lite": 1,
        "DSA-CI-Standard": 2,
        "DSA-Deep": 3,
        "DSA-Forensic": 4,
    }.get(profile, 0)


def _profile_from_enum(value: int) -> str:
    profiles = {
        1: "DSA-CI-Lite",
        2: "DSA-CI-Standard",
        3: "DSA-Deep",
        4: "DSA-Forensic",
    }
    try:
        return profiles[int(value)]
    except KeyError as exc:
        raise ValueError("unknown V3.3 evidence profile") from exc


def _encode_v33_chunks(
    records: list[CompactTokenEvidenceV3],
    base_k: int,
    cap: int,
) -> tuple[list[bytes], list[bytes]]:
    if cap < 1 or cap > 65535:
        raise ValueError("chunk_token_capacity must be between 1 and 65535 for V3.3")
    chunks = []
    payloads = []
    for chunk_index, start in enumerate(range(0, len(records), cap)):
        subset = records[start:start + cap]
        payload = bytearray()
        max_effective_topk = base_k
        counts = {TRIGGER_RANK: 0, TRIGGER_STOCHASTIC: 0, TRIGGER_HISTORY: 0, TRIGGER_CANARY: 0}
        for rec in subset:
            if rec.token_index != start + len(payloads) * 0 + (rec.token_index - start):
                raise ValueError("non-contiguous token evidence")
            max_effective_topk = max(max_effective_topk, rec.effective_topk)
            for bit in counts:
                counts[bit] += 1 if rec.trigger_flags & bit else 0
            record_flags = rec.record_flags
            chosen_rank = rec.chosen_rank
            if chosen_rank == 255:
                record_flags |= RECORD_HAS_FALLBACK_CHOSEN_ID
            payload += _TOKEN_V33_PREFIX.pack(rec.effective_topk - base_k, rec.chosen_rank, rec.trigger_flags, record_flags)
            if record_flags & RECORD_HAS_FALLBACK_CHOSEN_ID:
                payload += struct.pack("<I", rec.chosen_id)
            for token_id, score in zip(rec.topk_ids, rec.topk_scores):
                payload += _TOPK_V33.pack(int(token_id), quantize_score(score))
        payload_bytes = bytes(payload)
        payloads.append(payload_bytes)
        flags = 1 if start + len(subset) == len(records) else 0
        chunk = _CHUNK_V33.pack(
            chunk_index,
            start,
            len(subset),
            flags,
            base_k,
            max_effective_topk,
            counts[TRIGGER_RANK],
            counts[TRIGGER_STOCHASTIC],
            counts[TRIGGER_HISTORY],
            counts[TRIGGER_CANARY],
            len(payload_bytes),
            binascii.crc32(payload_bytes) & 0xFFFFFFFF,
            _sha256(payload_bytes),
        )
        chunks.append(chunk + payload_bytes)
    return chunks, payloads


def _encode_v33_spans(spans: list[SignalSpanEventV3], token_count: int) -> bytes:
    body = bytearray()
    for span in spans:
        if span.start_token >= span.end_token or span.end_token > token_count:
            raise ValueError("sparse span is outside token range")
        flags = SPAN_HAS_EVALUATOR_ID if span.evaluator_id is not None else 0
        body += _SPAN_V33.pack(span.start_token, span.end_token, span.signal_id, quantize_score(span.score), span.provenance_id)
        body += struct.pack("<B", flags)
        if span.evaluator_id is not None:
            body += _SPAN_V33_EVAL.pack(int(span.evaluator_id))
    return bytes(body)


def _pack_v33_manifest(
    artifact_hash: bytes,
    header_hash: bytes,
    chunk_root: bytes,
    token_count: int,
    chunk_count: int,
    span_count: int,
    raw_bytes: int,
    min_reconstructable: int,
    records: list[CompactTokenEvidenceV3],
    retention_requirement_hash: bytes,
) -> bytes:
    histogram = [0, 0, 0, 0]
    trigger_counts = [0, 0, 0, 0, 0, 0]
    bitmap = bytearray()
    for rec in records:
        if rec.effective_topk <= 3:
            histogram[0] += 1
        elif rec.effective_topk <= 5:
            histogram[1] += 1
        elif rec.effective_topk <= 10:
            histogram[2] += 1
        else:
            histogram[3] += 1
        bits = [TRIGGER_RANK, TRIGGER_STOCHASTIC, TRIGGER_HISTORY, TRIGGER_CANARY, TRIGGER_ESCALATION]
        active = 0
        for idx, bit in enumerate(bits):
            if rec.trigger_flags & bit:
                trigger_counts[idx] += 1
                active += 1
        if active > 1:
            trigger_counts[4] += 1
        bitmap.append(1 if rec.trigger_flags & TRIGGER_STOCHASTIC else 0)
    return _MANIFEST_V33.pack(
        artifact_hash,
        header_hash,
        chunk_root,
        token_count,
        chunk_count,
        span_count,
        raw_bytes,
        min_reconstructable,
        *histogram,
        *trigger_counts,
        _sha256(bytes(bitmap)),
        0,
        retention_requirement_hash,
    )


def _decode_tokens_and_spans(
    buf: bytes,
    *,
    pos: int,
    token_count: int,
    chunk_count: int,
    span_count: int,
    legacy_v31: bool,
) -> tuple[list[CompactTokenEvidenceV3], list[SignalSpanEventV3], int]:
    records: list[CompactTokenEvidenceV3] = []
    expected_start = 0
    for expected_chunk in range(chunk_count):
        if pos + _CHUNK.size > len(buf):
            raise ValueError("artifact is truncated in chunk header")
        chunk_index, start, count, byte_len, crc = _CHUNK.unpack_from(buf, pos)
        pos += _CHUNK.size
        if chunk_index != expected_chunk or start != expected_start:
            raise ValueError("compact chunks are missing or reordered")
        body = buf[pos:pos + byte_len]
        pos += byte_len
        if len(body) != byte_len or (binascii.crc32(body) & 0xFFFFFFFF) != crc:
            raise ValueError("compact chunk integrity check failed")
        bpos = 0
        for _ in range(count):
            if bpos + _TOKEN.size > len(body):
                raise ValueError("malformed token evidence")
            first, local_offset, eff = _TOKEN.unpack_from(body, bpos)
            bpos += _TOKEN.size
            fallback_chosen_id = None
            chosen_rank = 255
            legacy_chosen_id = None
            if legacy_v31:
                legacy_chosen_id = first
            elif first == 255:
                if bpos + 4 > len(body):
                    raise ValueError("malformed fallback chosen-id evidence")
                fallback_chosen_id = struct.unpack_from("<I", body, bpos)[0]
                bpos += 4
            else:
                chosen_rank = first
            token_index = start + local_offset
            if local_offset != len(records) - start or token_index != len(records):
                raise ValueError("missing, duplicate, or reordered token evidence")
            ids = []
            scores = []
            for _k in range(eff):
                if bpos + _TOPK.size > len(body):
                    raise ValueError("malformed top-k evidence")
                token_id, q_score = _TOPK.unpack_from(body, bpos)
                bpos += _TOPK.size
                ids.append(token_id)
                scores.append(dequantize_score(q_score))
            if legacy_v31:
                chosen_id = int(legacy_chosen_id)
                chosen_rank = ids.index(chosen_id) + 1 if chosen_id in ids else 255
            elif chosen_rank != 255:
                if chosen_rank < 1 or chosen_rank > len(ids):
                    raise ValueError("chosen rank is outside retained candidates")
                chosen_id = ids[chosen_rank - 1]
            else:
                chosen_id = int(fallback_chosen_id)
            records.append(CompactTokenEvidenceV3(token_index, chosen_id, tuple(ids), tuple(scores), eff, chosen_rank))
        if bpos != len(body):
            raise ValueError("malformed chunk payload")
        expected_start += count
    if len(records) != token_count:
        raise ValueError("missing token evidence")
    spans_out = []
    for _ in range(span_count):
        if pos + _SPAN.size > len(buf):
            raise ValueError("artifact is truncated in span events")
        start, end, signal_id, q_score = _SPAN.unpack_from(buf, pos)
        pos += _SPAN.size
        if start >= end or end > token_count:
            raise ValueError("sparse span is outside token range")
        spans_out.append(SignalSpanEventV3(start, end, signal_id, dequantize_score(q_score)))
    return records, spans_out, pos


def _decode_v31(buf: bytes) -> dict[str, Any]:
    if len(buf) < _HEADER_V31.size:
        raise ValueError("malformed compact evidence header for version 0x0301")
    magic, version, profile_len, meta_len, seq, token_count, cap, base_k, max_k, chunk_count, span_count = _HEADER_V31.unpack_from(buf, 0)
    pos = _HEADER_V31.size
    if profile_len == 0 or meta_len == 0 or pos + profile_len + meta_len > len(buf):
        raise ValueError("compact evidence layout mismatch for version 0x0301")
    try:
        profile_id = buf[pos:pos + profile_len].decode()
        pos += profile_len
        meta = json.loads(buf[pos:pos + meta_len].decode())
        pos += meta_len
    except Exception as exc:
        raise ValueError("compact evidence layout mismatch for version 0x0301") from exc
    try:
        get_evidence_profile(profile_id)
    except Exception as exc:
        raise ValueError("unknown compact evidence profile declaration") from exc
    if not isinstance(meta, dict) or meta.get("profile_id", profile_id) != profile_id:
        raise ValueError("malformed compact metadata")
    records, spans, pos = _decode_tokens_and_spans(buf, pos=pos, token_count=token_count, chunk_count=chunk_count, span_count=span_count, legacy_v31=True)
    if pos != len(buf):
        raise ValueError("unexpected trailing compact evidence bytes")
    digest = hashlib.sha256(buf).hexdigest()
    return {"header": MonologueSequenceHeaderV3(seq, token_count, profile_id, base_k, max_k, cap, schema_version=VERSION_V31), "tokens": records, "spans": spans, "meta": meta, "sha256": digest, "raw_bytes": len(buf)}


def _decode_v32(buf: bytes) -> dict[str, Any]:
    if len(buf) < _HEADER_V32.size:
        raise ValueError("malformed compact evidence header for version 0x0302")
    magic, version, profile_len, meta_len, seq, token_count, cap, base_k, max_k, chunk_count, span_count = _HEADER_V32.unpack_from(buf, 0)
    pos = _HEADER_V32.size
    if profile_len == 0 or meta_len == 0 or pos + profile_len + meta_len > len(buf):
        raise ValueError("malformed compact evidence header for version 0x0302")
    profile_id = buf[pos:pos + profile_len].decode()
    pos += profile_len
    try:
        get_evidence_profile(profile_id)
    except Exception as exc:
        raise ValueError("unknown compact evidence profile declaration") from exc
    try:
        meta = json.loads(buf[pos:pos + meta_len].decode())
    except Exception as exc:
        raise ValueError("malformed compact metadata") from exc
    pos += meta_len
    required_meta = {"evidence_profile", "assurance_class", "signal_schema_id", "signal_schema_hash", "probe_pack_id", "probe_pack_hash", "decoder_control_flags", "adaptive_policy_id", "verifier_budget_id", "retention_floor_policy_id", "quantization_id"}
    if not isinstance(meta, dict) or not required_meta.issubset(meta):
        raise ValueError("malformed compact metadata")
    if meta.get("profile_id", meta.get("evidence_profile")) != profile_id or meta.get("evidence_profile") != profile_id:
        raise ValueError("compact metadata profile declaration mismatch")
    if meta.get("assurance_class") != "DSA-R":
        raise ValueError("unsupported assurance class")
    records, spans, pos = _decode_tokens_and_spans(buf, pos=pos, token_count=token_count, chunk_count=chunk_count, span_count=span_count, legacy_v31=False)
    if pos != len(buf):
        raise ValueError("unexpected trailing compact evidence bytes")
    digest = hashlib.sha256(buf).hexdigest()
    return {"header": MonologueSequenceHeaderV3(seq, token_count, profile_id, base_k, max_k, cap, schema_version=VERSION_V32), "tokens": records, "spans": spans, "meta": meta, "sha256": digest, "raw_bytes": len(buf)}


def _decode_v33(buf: bytes) -> dict[str, Any]:
    if len(buf) < _HEADER_V33.size + 32 + _MANIFEST_V33.size:
        raise ValueError("malformed compact evidence header for version 0x0303")
    fields = _HEADER_V33.unpack_from(buf, 0)
    (
        magic,
        version,
        profile_enum,
        assurance_enum,
        prompt_nonce,
        sequence_id,
        tokenizer_id,
        signal_schema_id,
        signal_schema_hash,
        probe_pack_id,
        probe_pack_hash,
        decoder_control_flags,
        base_topk,
        max_adaptive_rank,
        adaptive_policy_id,
        stochastic_rate_ppm,
        audit_key_id,
        audit_selection_commitment_bytes,
        tension_map_id,
        tension_map_hash,
        quantization_id,
        chunk_token_capacity,
        verifier_work_profile_id,
        runtime_calibration_id,
        retention_policy_id,
        metadata_len,
        chunk_count,
        span_count,
        token_count,
    ) = fields
    if magic != MAGIC or version != VERSION_V33:
        raise ValueError("compact evidence schema mismatch")
    profile_id = _profile_from_enum(profile_enum)
    prof = get_evidence_profile(profile_id)
    if assurance_enum != 0:
        raise ValueError("software-only decoder supports DSA-R assurance class only")
    if base_topk != prof.base_k or not (1 <= base_topk <= max_adaptive_rank <= 255):
        raise ValueError("V3.3 profile/top-k declaration mismatch")
    if not (0 <= stochastic_rate_ppm <= 1_000_000):
        raise ValueError("invalid V3.3 stochastic sampling rate")
    pos = _HEADER_V33.size
    metadata_digest = buf[pos:pos + 32]
    pos += 32
    metadata_bytes = buf[pos:pos + metadata_len]
    pos += metadata_len
    if len(metadata_bytes) != metadata_len or _sha256(metadata_bytes) != metadata_digest:
        raise ValueError("V3.3 metadata digest mismatch")
    metadata = json.loads(metadata_bytes.decode("utf-8")) if metadata_bytes else {}
    if metadata.get("profile_id") != profile_id:
        raise ValueError("V3.3 metadata profile declaration mismatch")

    records: list[CompactTokenEvidenceV3] = []
    chunks: list[EvidenceChunkV3] = []
    chunk_payloads: list[bytes] = []
    expected_start = 0
    for expected_chunk in range(chunk_count):
        if pos + _CHUNK_V33.size > len(buf):
            raise ValueError("artifact is truncated in V3.3 chunk header")
        chunk_fields = _CHUNK_V33.unpack_from(buf, pos)
        pos += _CHUNK_V33.size
        (
            chunk_index,
            first_token_index,
            chunk_token_count,
            chunk_flags,
            chunk_base_topk,
            max_effective_topk,
            rank_count,
            stochastic_count,
            history_count,
            canary_count,
            payload_len,
            chunk_crc32,
            chunk_hash,
        ) = chunk_fields
        if chunk_index != expected_chunk or first_token_index != expected_start:
            raise ValueError("compact chunks are missing or reordered")
        payload = buf[pos:pos + payload_len]
        pos += payload_len
        if len(payload) != payload_len:
            raise ValueError("artifact is truncated in V3.3 chunk payload")
        if binascii.crc32(payload) & 0xFFFFFFFF != chunk_crc32 or _sha256(payload) != chunk_hash:
            raise ValueError("compact chunk integrity check failed")
        chunk_payloads.append(payload)
        chunks.append(EvidenceChunkV3(chunk_index, first_token_index, chunk_token_count, chunk_crc32))
        decoded_subset = _decode_v33_chunk_payload(payload, first_token_index, chunk_token_count, base_topk)
        if sum(1 for rec in decoded_subset if rec.trigger_flags & TRIGGER_RANK) != rank_count:
            raise ValueError("V3.3 rank trigger count mismatch")
        if sum(1 for rec in decoded_subset if rec.trigger_flags & TRIGGER_STOCHASTIC) != stochastic_count:
            raise ValueError("V3.3 stochastic trigger count mismatch")
        if sum(1 for rec in decoded_subset if rec.trigger_flags & TRIGGER_HISTORY) != history_count:
            raise ValueError("V3.3 history trigger count mismatch")
        if sum(1 for rec in decoded_subset if rec.trigger_flags & TRIGGER_CANARY) != canary_count:
            raise ValueError("V3.3 canary trigger count mismatch")
        records.extend(decoded_subset)
        expected_start += chunk_token_count
    if len(records) != token_count:
        raise ValueError("missing token evidence")

    spans: list[SignalSpanEventV3] = []
    for _ in range(span_count):
        if pos + _SPAN_V33.size + 1 > len(buf):
            raise ValueError("artifact is truncated in V3.3 span events")
        start, end, signal_id, q_score, provenance_id = _SPAN_V33.unpack_from(buf, pos)
        pos += _SPAN_V33.size
        flags = struct.unpack_from("<B", buf, pos)[0]
        pos += 1
        evaluator_id = None
        if flags & SPAN_HAS_EVALUATOR_ID:
            if pos + _SPAN_V33_EVAL.size > len(buf):
                raise ValueError("artifact is truncated in V3.3 span evaluator")
            evaluator_id = _SPAN_V33_EVAL.unpack_from(buf, pos)[0]
            pos += _SPAN_V33_EVAL.size
        if start >= end or end > token_count:
            raise ValueError("sparse span is outside token range")
        spans.append(SignalSpanEventV3(start, end, signal_id, dequantize_score(q_score), provenance_id, evaluator_id))

    if pos + _MANIFEST_V33.size != len(buf):
        raise ValueError("unexpected trailing or missing V3.3 manifest bytes")
    manifest_start = pos
    manifest_fields = _MANIFEST_V33.unpack_from(buf, manifest_start)
    manifest = _manifest_from_fields(manifest_fields)
    if manifest.token_count != token_count or manifest.chunk_count != chunk_count or manifest.span_event_count != span_count:
        raise ValueError("V3.3 manifest count mismatch")
    if manifest.sequence_header_hash != _sha256_hex(buf[:_HEADER_V33.size]):
        raise ValueError("V3.3 sequence header hash mismatch")
    if manifest.chunk_merkle_root != _merkle_root(chunk_payloads).hex():
        raise ValueError("V3.3 chunk Merkle root mismatch")
    preimage = buf[:manifest_start] + _manifest_with_zero_hash(buf[manifest_start:])
    if manifest.artifact_content_hash != _sha256_hex(preimage):
        raise ValueError("V3.3 artifact content hash mismatch")
    if manifest.raw_evidence_bytes != len(buf):
        raise ValueError("V3.3 raw evidence byte count mismatch")
    local_floor = v33_minimum_reconstructable_bytes(
        metadata_bytes=metadata_bytes,
        chunks=chunks,
        tokens=records,
        spans=spans,
    )
    if manifest.minimum_reconstructable_bytes != local_floor:
        raise ValueError("V3.3 minimum reconstructable byte count mismatch")
    return {
        "header": MonologueSequenceHeaderV3(sequence_id, token_count, profile_id, base_topk, max_adaptive_rank, chunk_token_capacity, schema_version=VERSION_V33),
        "tokens": records,
        "spans": spans,
        "meta": {
            **metadata,
            "audit_key_id": audit_key_id,
            "audit_selection_commitment": audit_selection_commitment_bytes.hex(),
            "commit_identity": metadata.get("commit_identity", ""),
            "policy_version": adaptive_policy_id,
            "stochastic_rate_ppm": stochastic_rate_ppm,
            "benchmark_id": metadata.get("benchmark_id", ""),
            "prompt_nonce": prompt_nonce,
            "pre_stochastic_eligibility_digest": metadata.get("pre_stochastic_eligibility_digest", ""),
        },
        "manifest": manifest,
        "sha256": hashlib.sha256(buf).hexdigest(),
        "raw_bytes": len(buf),
    }


def v33_minimum_reconstructable_bytes(
    *,
    metadata_bytes: bytes,
    chunks: list[EvidenceChunkV3],
    tokens: list[CompactTokenEvidenceV3],
    spans: list[SignalSpanEventV3],
) -> int:
    """Calculate the V3.3 floor solely from the decoded canonical wire layout."""
    token_bytes = sum(
        _TOKEN_V33_PREFIX.size
        + (_TOPK_V33.size * rec.effective_topk)
        + (4 if rec.record_flags & RECORD_HAS_FALLBACK_CHOSEN_ID else 0)
        for rec in tokens
    )
    span_bytes = sum(
        _SPAN_V33.size + 1 + (_SPAN_V33_EVAL.size if span.evaluator_id is not None else 0)
        for span in spans
    )
    return (
        _HEADER_V33.size
        + 32  # metadata digest
        + len(metadata_bytes)
        + len(chunks) * _CHUNK_V33.size
        + token_bytes
        + span_bytes
        + _MANIFEST_V33.size
    )


def _decode_v33_chunk_payload(payload: bytes, start: int, count: int, base_topk: int) -> list[CompactTokenEvidenceV3]:
    records = []
    pos = 0
    for offset in range(count):
        if pos + _TOKEN_V33_PREFIX.size > len(payload):
            raise ValueError("malformed V3.3 token evidence")
        delta, chosen_rank_raw, trigger_flags, record_flags = _TOKEN_V33_PREFIX.unpack_from(payload, pos)
        pos += _TOKEN_V33_PREFIX.size
        effective_topk = base_topk + delta
        chosen_id = None
        if record_flags & RECORD_HAS_FALLBACK_CHOSEN_ID:
            if pos + 4 > len(payload):
                raise ValueError("malformed V3.3 fallback chosen-id evidence")
            chosen_id = struct.unpack_from("<I", payload, pos)[0]
            pos += 4
        ids = []
        scores = []
        for _ in range(effective_topk):
            if pos + _TOPK_V33.size > len(payload):
                raise ValueError("malformed V3.3 top-k evidence")
            token_id, q_score = _TOPK_V33.unpack_from(payload, pos)
            pos += _TOPK_V33.size
            ids.append(token_id)
            scores.append(dequantize_score(q_score))
        if chosen_id is None:
            chosen_rank = int(chosen_rank_raw)
            if chosen_rank < 1 or chosen_rank > len(ids):
                raise ValueError("V3.3 chosen rank is outside retained candidates")
            chosen_id = ids[chosen_rank - 1]
        else:
            chosen_rank = ids.index(chosen_id) + 1 if chosen_id in ids else 255
        records.append(
            CompactTokenEvidenceV3(
                token_index=start + offset,
                chosen_id=chosen_id,
                topk_ids=tuple(ids),
                topk_scores=tuple(scores),
                effective_topk=effective_topk,
                chosen_rank=chosen_rank,
                trigger_flags=trigger_flags,
                record_flags=record_flags,
            )
        )
    if pos != len(payload):
        raise ValueError("malformed V3.3 chunk payload")
    return records


def _manifest_from_fields(fields: tuple[Any, ...]) -> EvidenceManifestV33:
    histogram = {3: fields[8], 5: fields[9], 10: fields[10], 255: fields[11]}
    trigger_counts = dict(zip(_TRIGGER_NAMES, fields[12:18]))
    return EvidenceManifestV33(
        artifact_content_hash=fields[0].hex(),
        sequence_header_hash=fields[1].hex(),
        chunk_merkle_root=fields[2].hex(),
        token_count=fields[3],
        chunk_count=fields[4],
        span_event_count=fields[5],
        raw_evidence_bytes=fields[6],
        minimum_reconstructable_bytes=fields[7],
        effective_k_histogram=histogram,
        trigger_count_by_reason=trigger_counts,
        audit_selection_digest=fields[18].hex(),
        local_audit_outcome="LOCAL_PASS" if fields[19] == 0 else "REVIEW",
        retention_requirement_hash=fields[20].hex(),
    )


def _manifest_with_zero_hash(manifest: bytes) -> bytes:
    return ZERO_HASH + manifest[32:]


def decode_compact_sequence(buf: bytes) -> dict[str, Any]:
    if len(buf) < PREFIX.size:
        raise ValueError("malformed compact evidence header")
    magic, version = PREFIX.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("compact evidence schema mismatch")
    if version == VERSION_V31:
        return _decode_v31(buf)
    if version == VERSION_V32:
        return _decode_v32(buf)
    if version == VERSION_V33:
        return _decode_v33(buf)
    raise ValueError(f"unsupported compact evidence version 0x{version:04x}")


def verify_keyed_replay(decoded: dict[str, Any] | bytes, audit_keys: dict[int, bytes]) -> None:
    data = decode_compact_sequence(decoded) if isinstance(decoded, (bytes, bytearray)) else decoded
    if data["header"].schema_version != VERSION_V33:
        return
    meta = data.get("meta", {})
    audit_key_id = int(meta.get("audit_key_id", 0))
    if audit_key_id not in audit_keys:
        raise ValueError("unknown audit key id")
    key = audit_keys[audit_key_id]
    records = data["tokens"]
    base_k = int(data["header"].base_k)
    recomputed_identity = _commit_identity(records, base_k)
    serialized_identity = str(meta.get("commit_identity", ""))
    if not hmac.compare_digest(recomputed_identity, serialized_identity):
        raise ValueError("serialized commit identity mismatch")

    # Rank is independently derivable from retained evidence. History/canary
    # provenance validation remains Phase 2, but their exact encode-time state is
    # authenticated by the keyed commitment in Phase 1.
    for rec in records:
        expected_rank = rec.chosen_rank > base_k and rec.chosen_rank <= int(data["header"].max_adaptive_k)
        if bool(rec.trigger_flags & TRIGGER_RANK) != expected_rank:
            raise ValueError("pre-stochastic rank eligibility mismatch")
    eligibility_digest = _pre_stochastic_eligibility_digest(records)
    serialized_eligibility = str(meta.get("pre_stochastic_eligibility_digest", ""))
    if not hmac.compare_digest(eligibility_digest, serialized_eligibility):
        raise ValueError("pre-stochastic eligibility mismatch")

    prompt_nonce = int(meta.get("prompt_nonce", -1))
    if prompt_nonce < 0 or prompt_nonce > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("missing or invalid 64-bit prompt nonce")
    expected_commitment = audit_selection_commitment(
        key,
        commit_identity=recomputed_identity,
        sequence_id=prompt_nonce,
        policy_version=int(meta.get("policy_version", 210)),
        rate_ppm=int(meta.get("stochastic_rate_ppm", 0)),
        benchmark_id=str(meta.get("benchmark_id", "")),
        eligibility_digest=eligibility_digest,
    ).hex()
    if not hmac.compare_digest(expected_commitment, str(meta.get("audit_selection_commitment", ""))):
        raise ValueError("audit selection commitment mismatch")
    for rec in records:
        otherwise_untriggered = not (rec.trigger_flags & (TRIGGER_RANK | TRIGGER_HISTORY | TRIGGER_CANARY))
        expected = False
        if otherwise_untriggered and int(meta.get("stochastic_rate_ppm", 0)):
            expected = keyed_sample_selected(
                key,
                commit_identity=recomputed_identity,
                sequence_id=prompt_nonce,
                token_index=rec.token_index,
                policy_version=int(meta.get("policy_version", 210)),
                rate_ppm=int(meta.get("stochastic_rate_ppm", 0)),
                benchmark_id=str(meta.get("benchmark_id", "")),
            )
        observed = bool(rec.trigger_flags & TRIGGER_STOCHASTIC)
        if observed != expected:
            raise ValueError("keyed stochastic selection replay mismatch")


def reconstruct_token_evidence(buf_or_decoded: bytes | dict[str, Any]) -> list[dict[str, Any]]:
    decoded = decode_compact_sequence(buf_or_decoded) if isinstance(buf_or_decoded, (bytes, bytearray)) else buf_or_decoded
    spans = decoded.get("spans", [])
    out = []
    for rec in decoded["tokens"]:
        active = [span for span in spans if span.start_token <= rec.token_index < span.end_token]
        out.append({
            "token_index": rec.token_index,
            "chosen_id": rec.chosen_id,
            "topk_ids": list(rec.topk_ids),
            "topk_scores": list(rec.topk_scores),
            "effective_topk": rec.effective_topk,
            "chosen_rank": rec.chosen_rank,
            "trigger_flags": rec.trigger_flags,
            "record_flags": rec.record_flags,
            "signals": active,
        })
    return out
