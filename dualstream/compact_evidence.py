from __future__ import annotations

import binascii
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Iterable

from .evidence_profile import get_evidence_profile

MAGIC = b"DSAEV29\0"
VERSION = 0x0301
SCORE_SCALE = 255.0
SCORE_TOLERANCE = 0.5 / SCORE_SCALE
_HEADER = struct.Struct("<8sHBBQIIHHHH")
_CHUNK = struct.Struct("<IIIII")
_TOKEN = struct.Struct("<IBB")
_TOPK = struct.Struct("<IB")
_SPAN = struct.Struct("<IIHB")


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


@dataclass(frozen=True)
class SignalSpanEventV3:
    start_token: int
    end_token: int
    signal_id: int
    score: float


def quantize_score(score: float) -> int:
    return max(0, min(255, int(round(float(score) * SCORE_SCALE))))


def dequantize_score(raw: int) -> float:
    return int(raw) / SCORE_SCALE


def choose_effective_topk(chosen_id: int, candidate_ids: list[int] | tuple[int, ...], base_k: int, max_adaptive_k: int, adaptive: bool = True) -> int:
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


def _normalise_tokens(tokens: Iterable[Any], base_k: int, max_adaptive_k: int, adaptive: bool) -> list[CompactTokenEvidenceV3]:
    out: list[CompactTokenEvidenceV3] = []
    for idx, rec in enumerate(tokens):
        if isinstance(rec, CompactTokenEvidenceV3):
            out.append(rec)
            continue
        token_index = int(getattr(rec, "token_index", idx) if not isinstance(rec, dict) else rec.get("token_index", idx))
        chosen_id = int(getattr(rec, "chosen_id") if not isinstance(rec, dict) else rec["chosen_id"])
        topk = getattr(rec, "topk", None) if not isinstance(rec, dict) else rec.get("topk")
        if topk is not None:
            ids = [int(t["token_id"] if isinstance(t, dict) else getattr(t, "token_id")) for t in topk]
            scores = [float(t["prob"] if isinstance(t, dict) else getattr(t, "prob")) for t in topk]
        else:
            ids = [int(x) for x in (rec["topk_ids"] if isinstance(rec, dict) else getattr(rec, "topk_ids"))]
            scores = [float(x) for x in (rec["topk_scores"] if isinstance(rec, dict) else getattr(rec, "topk_scores"))]
        eff = int((rec.get("effective_topk") if isinstance(rec, dict) else getattr(rec, "effective_topk", 0)) or choose_effective_topk(chosen_id, ids, base_k, max_adaptive_k, adaptive))
        eff = min(eff, len(ids), max_adaptive_k)
        out.append(CompactTokenEvidenceV3(token_index, chosen_id, tuple(ids[:eff]), tuple(scores[:eff]), eff))
    return out


def encode_compact_sequence(tokens: Iterable[Any], *, profile: str = "DSA-CI-Lite", sequence_id: int = 0, chunk_token_capacity: int = 256, adaptive_k: bool = True, max_adaptive_k: int | None = None, spans: Iterable[SignalSpanEventV3 | dict[str, Any]] = ()) -> bytes:
    prof = get_evidence_profile(profile)
    max_k = prof.max_adaptive_k if max_adaptive_k is None else int(max_adaptive_k)
    records = _normalise_tokens(tokens, prof.base_k, max_k, adaptive_k)
    spans_norm = [s if isinstance(s, SignalSpanEventV3) else SignalSpanEventV3(int(s["start_token"]), int(s["end_token"]), int(s["signal_id"]), float(s["score"])) for s in spans]
    meta = json.dumps({"profile_id": prof.profile_id.value, "verifier_budget_id": prof.verifier_budget_id}, sort_keys=True, separators=(",", ":")).encode()
    chunks: list[bytes] = []
    cap = max(1, int(chunk_token_capacity))
    if cap > 256:
        raise ValueError("chunk_token_capacity must be between 1 and 256 for compact token offsets")
    for chunk_index, start in enumerate(range(0, len(records), cap)):
        subset = records[start:start + cap]
        body = bytearray()
        for local_offset, r in enumerate(subset):
            expected_token_index = start + local_offset
            if r.token_index != expected_token_index:
                raise ValueError(f"token evidence index {r.token_index} does not match expected {expected_token_index}")
            body += _TOKEN.pack(r.chosen_id, local_offset, r.effective_topk)
            for tid, score in zip(r.topk_ids, r.topk_scores):
                body += _TOPK.pack(int(tid), quantize_score(score))
        crc = binascii.crc32(body) & 0xFFFFFFFF
        chunks.append(_CHUNK.pack(chunk_index, start, len(subset), len(body), crc) + body)
    span_body = bytearray()
    for s in spans_norm:
        span_body += _SPAN.pack(s.start_token, s.end_token, s.signal_id, quantize_score(s.score))
    header = _HEADER.pack(MAGIC, VERSION, len(prof.profile_id.value), len(meta), int(sequence_id) & 0xFFFFFFFFFFFFFFFF, len(records), cap, prof.base_k, max_k, len(chunks), len(spans_norm))
    return bytes(header + prof.profile_id.value.encode() + meta + b"".join(chunks) + span_body)


def decode_compact_sequence(buf: bytes) -> dict[str, Any]:
    if len(buf) < _HEADER.size:
        raise ValueError("artifact is truncated before compact header")
    magic, version, profile_len, meta_len, seq, token_count, cap, base_k, max_k, chunk_count, span_count = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError("compact evidence schema mismatch")
    pos = _HEADER.size
    profile_id = buf[pos:pos + profile_len].decode(); pos += profile_len
    meta = json.loads(buf[pos:pos + meta_len].decode()); pos += meta_len
    records: list[CompactTokenEvidenceV3] = []
    expected_start = 0
    for expected_chunk in range(chunk_count):
        if pos + _CHUNK.size > len(buf):
            raise ValueError("artifact is truncated in chunk header")
        chunk_index, start, count, byte_len, crc = _CHUNK.unpack_from(buf, pos); pos += _CHUNK.size
        if chunk_index != expected_chunk or start != expected_start:
            raise ValueError("compact chunks are missing or reordered")
        body = buf[pos:pos + byte_len]; pos += byte_len
        if len(body) != byte_len or (binascii.crc32(body) & 0xFFFFFFFF) != crc:
            raise ValueError("compact chunk integrity check failed")
        bpos = 0
        for _ in range(count):
            if bpos + _TOKEN.size > len(body):
                raise ValueError("malformed token evidence")
            chosen_id, local_offset, eff = _TOKEN.unpack_from(body, bpos); bpos += _TOKEN.size
            token_index = start + local_offset
            if local_offset != len(records) - start or token_index != len(records):
                raise ValueError("missing, duplicate, or reordered token evidence")
            ids=[]; scores=[]
            for _k in range(eff):
                if bpos + _TOPK.size > len(body):
                    raise ValueError("malformed top-k evidence")
                tid, q = _TOPK.unpack_from(body, bpos); bpos += _TOPK.size
                ids.append(tid); scores.append(dequantize_score(q))
            records.append(CompactTokenEvidenceV3(token_index, chosen_id, tuple(ids), tuple(scores), eff))
        if bpos != len(body):
            raise ValueError("malformed chunk payload")
        expected_start += count
    if len(records) != token_count:
        raise ValueError("missing token evidence")
    spans=[]
    for _ in range(span_count):
        if pos + _SPAN.size > len(buf):
            raise ValueError("artifact is truncated in span events")
        start, end, sid, q = _SPAN.unpack_from(buf, pos); pos += _SPAN.size
        if start > end or end > token_count:
            raise ValueError("sparse span is outside token range")
        spans.append(SignalSpanEventV3(start, end, sid, dequantize_score(q)))
    if pos != len(buf):
        raise ValueError("unexpected trailing compact evidence bytes")
    digest = hashlib.sha256(buf).hexdigest()
    return {"header": MonologueSequenceHeaderV3(seq, token_count, profile_id, base_k, max_k, cap), "tokens": records, "spans": spans, "meta": meta, "sha256": digest, "raw_bytes": len(buf)}


def reconstruct_token_evidence(buf_or_decoded: bytes | dict[str, Any]) -> list[dict[str, Any]]:
    decoded = decode_compact_sequence(buf_or_decoded) if isinstance(buf_or_decoded, (bytes, bytearray)) else buf_or_decoded
    spans = decoded.get("spans", [])
    out=[]
    for r in decoded["tokens"]:
        active = [s for s in spans if s.start_token <= r.token_index < s.end_token]
        out.append({"token_index": r.token_index, "chosen_id": r.chosen_id, "topk_ids": list(r.topk_ids), "topk_scores": list(r.topk_scores), "effective_topk": r.effective_topk, "signals": active})
    return out
