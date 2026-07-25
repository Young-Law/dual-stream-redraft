from __future__ import annotations

from dataclasses import dataclass
from .compact_evidence import decode_compact_sequence, compute_v33_local_minimum_reconstructable_bytes, _HEADER, _CHUNK, _TOKEN, _TOPK, _SPAN
from .evidence_profile import get_evidence_profile


@dataclass(frozen=True)
class EvidenceBudgetSummary:
    profile_id: str
    token_count: int
    raw_bytes: int
    raw_bytes_per_token: float
    ceiling_bytes_per_token: int
    minimum_reconstructable_bytes: int
    retention_floor_margin: int
    retained_reconstructable_bytes: int
    compressed_bytes_per_token: float | None = None


def compute_minimum_reconstructable_bytes(token_count: int, effective_topks: list[int], profile: str = "DSA-CI-Lite", *, chunk_count: int = 1, profile_metadata_bytes: int = 0, fallback_count: int = 0, span_count: int = 0, profile_id_bytes: int = 0) -> int:
    get_evidence_profile(profile)
    token_floor = sum(_TOKEN.size + max(0, int(k)) * _TOPK.size for k in effective_topks[:token_count])
    return _HEADER.size + profile_id_bytes + profile_metadata_bytes + chunk_count * _CHUNK.size + token_floor + fallback_count * 4 + span_count * _SPAN.size


def compute_evidence_budget_summary(artifact: bytes | str, profile: str = "DSA-CI-Lite") -> EvidenceBudgetSummary:
    data = artifact if isinstance(artifact, (bytes, bytearray)) else open(str(artifact), "rb").read()
    decoded = decode_compact_sequence(bytes(data))
    prof = get_evidence_profile(profile)
    token_count = len(decoded["tokens"])
    if token_count <= 0:
        raise ValueError("summary-only artifact has no reconstructable token evidence")
    eff = [int(t.effective_topk) for t in decoded["tokens"]]
    header = decoded["header"]
    manifest = decoded.get("manifest")
    if manifest is not None and getattr(header, "schema_version", None) == 0x0303:
<<<<<<< HEAD
        # The V3.3 decoder has independently reconstructed the exact binary-layout
        # floor and rejected any manifest mismatch before returning this value.
        floor = len(data)
=======
        floor = compute_v33_local_minimum_reconstructable_bytes(bytes(data))
        if int(manifest.minimum_reconstructable_bytes) != floor:
            raise ValueError("V3.3 minimum reconstructable byte floor mismatch")
>>>>>>> f2bdf45 (Authenticate complete V3.3 replay context)
    else:
        fallback_count = sum(1 for t in decoded["tokens"] if int(getattr(t, "chosen_rank", 255)) == 255)
        meta_len = len(__import__("json").dumps(decoded.get("meta", {}), sort_keys=True, separators=(",", ":")).encode())
        chunk_count = (token_count + int(header.chunk_token_capacity) - 1) // int(header.chunk_token_capacity)
        floor = compute_minimum_reconstructable_bytes(token_count, eff, prof.profile_id.value, chunk_count=chunk_count, profile_metadata_bytes=meta_len, fallback_count=fallback_count, span_count=len(decoded.get("spans", [])), profile_id_bytes=len(header.profile_id.encode()))
    raw = len(data)
    return EvidenceBudgetSummary(prof.profile_id.value, token_count, raw, raw / token_count, prof.ceiling_bytes_per_token, floor, raw - floor, raw)


def assert_evidence_budget(summary: EvidenceBudgetSummary) -> None:
    if summary.raw_bytes_per_token > summary.ceiling_bytes_per_token:
        raise ValueError(f"raw bytes/token {summary.raw_bytes_per_token:.3f} exceeds ceiling {summary.ceiling_bytes_per_token}")


def assert_retention_floor(summary: EvidenceBudgetSummary) -> None:
    if summary.retained_reconstructable_bytes < summary.minimum_reconstructable_bytes:
        raise ValueError("retained compact evidence is below the reconstructable floor")
    if summary.token_count <= 0:
        raise ValueError("summary-only artifact has no reconstructable token evidence")
