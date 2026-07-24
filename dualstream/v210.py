"""Phase-1 DSA v2.10 helpers.

The normative V3.3 artifact wire format lives in :mod:`dualstream.compact_evidence`
as a compact binary extension of the existing evidence family. This module keeps
small, typed helper APIs for keyed replay and canonical hashing without defining
a separate JSON artifact format.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Any

from .compact_evidence import (
    TRIGGER_CANARY,
    TRIGGER_HISTORY,
    TRIGGER_RANK,
    TRIGGER_STOCHASTIC,
    VERSION_V33,
    audit_selection_commitment,
    decode_compact_sequence,
    encode_compact_sequence,
    keyed_sample_selected,
    verify_keyed_replay,
)

WIRE_VERSION_V33 = VERSION_V33


def sha256_hex(data: bytes | str) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def sign_hmac(payload: bytes, key: bytes, signer_id: str) -> dict[str, str]:
    return {
        "alg": "HMAC-SHA256",
        "signer_id": signer_id,
        "signature": hmac.new(key, payload, hashlib.sha256).hexdigest(),
    }


def verify_hmac(payload: bytes, signature: dict[str, str], keys: dict[str, bytes]) -> bool:
    signer_id = str(signature.get("signer_id", ""))
    key = keys.get(signer_id)
    if key is None:
        return False
    expected = sign_hmac(payload, key, signer_id)["signature"]
    return hmac.compare_digest(expected, str(signature.get("signature", "")))


def build_v33_artifact(tokens: list[dict[str, Any]], **kwargs: Any) -> bytes:
    return encode_compact_sequence(tokens, wire_version=VERSION_V33, **kwargs)


def decode_v33(data: bytes) -> dict[str, Any]:
    decoded = decode_compact_sequence(data)
    if decoded["header"].schema_version != VERSION_V33:
        raise ValueError("not a V3.3 artifact")
    return decoded


def verify_v33(data: bytes, *, audit_keys: dict[int, bytes] | None = None) -> dict[str, Any]:
    decoded = decode_v33(data)
    if audit_keys is not None:
        verify_keyed_replay(decoded, audit_keys)
    return {
        "outcome": "LOCAL_PASS",
        "errors": [],
        "manifest": decoded.get("manifest"),
        "token_count": decoded["header"].token_count,
    }
