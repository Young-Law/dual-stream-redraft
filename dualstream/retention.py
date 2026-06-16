from __future__ import annotations

from dataclasses import dataclass
from .compact_evidence import decode_compact_sequence
from .evidence_profile import get_evidence_profile


@dataclass(frozen=True)
class EvidenceBudgetSummary:
    profile_id: str
    token_count: int
    raw_bytes: int
    raw_bytes_per_token: float
    ceiling_bytes_per_token: int
    minimum_reconstructable_bytes: int
    retention_floor_margin: float


def compute_minimum_reconstructable_bytes(token_count: int, effective_topks: list[int], profile: str = "DSA-CI-Lite") -> int:
    prof = get_evidence_profile(profile)
    return sum(prof.minimum_reconstructable_bytes_per_token(k) for k in effective_topks[:token_count])


def compute_evidence_budget_summary(artifact: bytes | str, profile: str = "DSA-CI-Lite") -> EvidenceBudgetSummary:
    data = artifact if isinstance(artifact, (bytes, bytearray)) else open(str(artifact), "rb").read()
    decoded = decode_compact_sequence(bytes(data))
    prof = get_evidence_profile(profile)
    token_count = len(decoded["tokens"])
    if token_count <= 0:
        raise ValueError("summary-only artifact has no reconstructable token evidence")
    eff = [int(t.effective_topk) for t in decoded["tokens"]]
    floor = compute_minimum_reconstructable_bytes(token_count, eff, prof)
    raw = len(data)
    return EvidenceBudgetSummary(prof.profile_id.value, token_count, raw, raw / token_count, prof.ceiling_bytes_per_token, floor, raw - floor)


def assert_evidence_budget(summary: EvidenceBudgetSummary) -> None:
    if summary.raw_bytes_per_token > summary.ceiling_bytes_per_token:
        raise ValueError(f"raw bytes/token {summary.raw_bytes_per_token:.3f} exceeds ceiling {summary.ceiling_bytes_per_token}")


def assert_retention_floor(summary: EvidenceBudgetSummary) -> None:
    if summary.raw_bytes < summary.minimum_reconstructable_bytes:
        raise ValueError("retained compact evidence is below the reconstructable floor")
    if summary.token_count <= 0:
        raise ValueError("summary-only artifact has no reconstructable token evidence")
