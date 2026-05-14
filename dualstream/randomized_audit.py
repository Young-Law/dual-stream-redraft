from __future__ import annotations
import hashlib, json, random
from typing import Any, Dict


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def randomized_selection(nonce: int | None, *, policy_id: str = "rand-v1", manifest_data: Any = None, total_heads: int = 16, subset_size: int = 4, sequence_id: str | None = None, token_index: int | None = None) -> Dict[str, Any]:
    nonce = 0 if nonce is None else int(nonce)
    manifest_hash = _stable_hash(manifest_data)
    seed_material = f"{nonce}|{policy_id}|{manifest_hash}|{sequence_id or ''}|{token_index if token_index is not None else ''}"
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    heads = sorted(rng.sample(list(range(total_heads)), k=min(subset_size, total_heads)))
    path = f"path-{seed % 13}"
    return {
        "audit_nonce_hash": hashlib.sha256(str(nonce).encode()).hexdigest()[:16],
        "randomized_policy_id": policy_id,
        "randomized_manifest_hash": manifest_hash,
        "audit_path_id": path,
        "randomized_probe_selection": {"heads": heads, "subset_size": subset_size, "path_ids": [path]},
    }


def verify_randomized_selection(selection: Dict[str, Any], *, nonce: int | None, policy_id: str = "rand-v1", manifest_data: Any = None, total_heads: int = 16, subset_size: int = 4, sequence_id: str | None = None, token_index: int | None = None) -> bool:
    expected = randomized_selection(nonce, policy_id=policy_id, manifest_data=manifest_data, total_heads=total_heads, subset_size=subset_size, sequence_id=sequence_id, token_index=token_index)
    return selection == expected
