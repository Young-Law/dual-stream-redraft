from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import binascii
import struct

import numpy as np

# Appendix C (v2.2 redraft): conceptual schema for MonologueFrameV1.
# This module provides:
# - a concrete dataclass representation
# - a binary codec that matches the conceptual struct layout
# - optional CRC32 (frame-level integrity)

MAGIC_MONO = 0x4D4F4E4F  # 'MONO' in ASCII, big-endian bytes; stored as uint32 little-endian.
VERSION_V1 = 0x0001


@dataclass(frozen=True)
class TopKToken:
    token_id: int
    prob: float  # stored as float16 in binary


@dataclass(frozen=True)
class AttnSummary:
    layer: int
    head: int
    top_token_idx: int
    weight: float  # stored as float16 in binary


@dataclass(frozen=True)
class ConceptScore:
    concept_id: int
    score: float  # stored as float16 in binary


@dataclass
class MonologueFrameV1:
    """
    Evidence frame emitted 1:1 with Answer tokens.

    Fields match Appendix C. Additional host metadata (sequence id, timestamps, etc.) should be tracked
    outside the frame or via a higher-level envelope.

    If `crc32` is present, it MUST be computed over the frame bytes up to (but not including) the crc32
    field, using standard IEEE CRC-32.
    """
    prompt_nonce: int
    token_index: int
    chosen_id: int
    topk: List[TopKToken]

    attn: List[AttnSummary] = field(default_factory=list)
    concepts: List[ConceptScore] = field(default_factory=list)
    signal_schema_id: Optional[str] = None
    signal_schema_hash: Optional[str] = None
    probe_pack_id: Optional[str] = None
    probe_pack_hash: Optional[str] = None
    capture_stage: Optional[str] = None
    decode_controls_applied: Optional[List[str]] = None

    crc32: Optional[int] = None  # optional in the paper

    magic: int = MAGIC_MONO
    version: int = VERSION_V1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magic": self.magic,
            "version": self.version,
            "prompt_nonce": int(self.prompt_nonce),
            "token_index": int(self.token_index),
            "chosen_id": int(self.chosen_id),
            "topk": [{"token_id": int(t.token_id), "prob": float(t.prob)} for t in self.topk],
            "attn": [
                {
                    "layer": int(a.layer),
                    "head": int(a.head),
                    "top_token_idx": int(a.top_token_idx),
                    "weight": float(a.weight),
                }
                for a in self.attn
            ],
            "concepts": [{"concept_id": int(c.concept_id), "score": float(c.score)} for c in self.concepts],
            "signal_schema_id": self.signal_schema_id,
            "signal_schema_hash": self.signal_schema_hash,
            "probe_pack_id": self.probe_pack_id,
            "probe_pack_hash": self.probe_pack_hash,
            "capture_stage": self.capture_stage,
            "decode_controls_applied": self.decode_controls_applied,
            "crc32": None if self.crc32 is None else int(self.crc32),
        }


def _f16_pack(x: float) -> bytes:
    # Enforce little-endian float16.
    return np.asarray(x, dtype=np.dtype("<f2")).tobytes()


def _f16_unpack(b: bytes) -> float:
    return float(np.frombuffer(b, dtype=np.dtype("<f2"))[0])


def encode_frame(frame: MonologueFrameV1, include_crc32: bool = True) -> bytes:
    """
    Binary encoding that follows Appendix C's conceptual struct ordering.

    Layout (little-endian):
      uint32 magic
      uint16 version
      uint64 prompt_nonce
      uint32 token_index
      uint32 chosen_id
      uint16 topk
      K * (uint32 token_id, float16 prob)
      uint16 attn_count
      attn_count * (uint16 layer, uint16 head, uint16 top_token_idx, float16 weight)
      uint16 concept_count
      concept_count * (uint16 concept_id, float16 score)
      uint32 crc32   (optional)
    """
    if frame.magic != MAGIC_MONO:
        raise ValueError(f"Bad magic: {frame.magic:#x}")
    if frame.version != VERSION_V1:
        raise ValueError(f"Unsupported version: {frame.version:#x}")

    parts: List[bytes] = []
    parts.append(struct.pack("<I", frame.magic))
    parts.append(struct.pack("<H", frame.version))
    parts.append(struct.pack("<Q", int(frame.prompt_nonce) & 0xFFFFFFFFFFFFFFFF))
    parts.append(struct.pack("<I", int(frame.token_index) & 0xFFFFFFFF))
    parts.append(struct.pack("<I", int(frame.chosen_id) & 0xFFFFFFFF))

    k = len(frame.topk)
    if k > 65535:
        raise ValueError("topk too large")
    parts.append(struct.pack("<H", k))
    for t in frame.topk:
        parts.append(struct.pack("<I", int(t.token_id) & 0xFFFFFFFF))
        parts.append(_f16_pack(float(t.prob)))

    attn_count = len(frame.attn)
    if attn_count > 65535:
        raise ValueError("attn_count too large")
    parts.append(struct.pack("<H", attn_count))
    for a in frame.attn:
        parts.append(struct.pack("<H", int(a.layer) & 0xFFFF))
        parts.append(struct.pack("<H", int(a.head) & 0xFFFF))
        parts.append(struct.pack("<H", int(a.top_token_idx) & 0xFFFF))
        parts.append(_f16_pack(float(a.weight)))

    concept_count = len(frame.concepts)
    if concept_count > 65535:
        raise ValueError("concept_count too large")
    parts.append(struct.pack("<H", concept_count))
    for c in frame.concepts:
        parts.append(struct.pack("<H", int(c.concept_id) & 0xFFFF))
        parts.append(_f16_pack(float(c.score)))

    payload = b"".join(parts)

    if include_crc32:
        crc = binascii.crc32(payload) & 0xFFFFFFFF
        payload += struct.pack("<I", crc)
    return payload


def decode_frame(buf: bytes, require_crc32: bool = False) -> MonologueFrameV1:
    """
    Decode a frame encoded by `encode_frame`.

    If `require_crc32=True`, the last 4 bytes MUST be present and match.
    If `require_crc32=False`, crc32 will be parsed iff remaining bytes indicate it is present.
    """
    mv = memoryview(buf)
    off = 0

    def take(n: int) -> bytes:
        nonlocal off
        if off + n > len(mv):
            raise ValueError("Truncated frame")
        b = mv[off : off + n].tobytes()
        off += n
        return b

    magic = struct.unpack("<I", take(4))[0]
    version = struct.unpack("<H", take(2))[0]
    if magic != MAGIC_MONO:
        raise ValueError(f"Bad magic: {magic:#x}")
    if version != VERSION_V1:
        raise ValueError(f"Unsupported version: {version:#x}")

    prompt_nonce = struct.unpack("<Q", take(8))[0]
    token_index = struct.unpack("<I", take(4))[0]
    chosen_id = struct.unpack("<I", take(4))[0]
    k = struct.unpack("<H", take(2))[0]

    topk: List[TopKToken] = []
    for _ in range(k):
        tid = struct.unpack("<I", take(4))[0]
        prob = _f16_unpack(take(2))
        topk.append(TopKToken(token_id=tid, prob=prob))

    attn_count = struct.unpack("<H", take(2))[0]
    attn: List[AttnSummary] = []
    for _ in range(attn_count):
        layer = struct.unpack("<H", take(2))[0]
        head = struct.unpack("<H", take(2))[0]
        top_token_idx = struct.unpack("<H", take(2))[0]
        weight = _f16_unpack(take(2))
        attn.append(AttnSummary(layer=layer, head=head, top_token_idx=top_token_idx, weight=weight))

    concept_count = struct.unpack("<H", take(2))[0]
    concepts: List[ConceptScore] = []
    for _ in range(concept_count):
        cid = struct.unpack("<H", take(2))[0]
        score = _f16_unpack(take(2))
        concepts.append(ConceptScore(concept_id=cid, score=score))

    remaining = len(mv) - off
    crc32: Optional[int] = None
    if remaining == 4:
        crc32 = struct.unpack("<I", take(4))[0]
        computed = binascii.crc32(mv[: len(mv) - 4].tobytes()) & 0xFFFFFFFF
        if crc32 != computed:
            raise ValueError(f"CRC32 mismatch: got {crc32:#x}, computed {computed:#x}")
    elif remaining == 0:
        if require_crc32:
            raise ValueError("Missing crc32")
    else:
        raise ValueError(f"Unexpected trailing bytes: {remaining}")

    return MonologueFrameV1(
        magic=magic,
        version=version,
        prompt_nonce=prompt_nonce,
        token_index=token_index,
        chosen_id=chosen_id,
        topk=topk,
        attn=attn,
        concepts=concepts,
        crc32=crc32,
    )
