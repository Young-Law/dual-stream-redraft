from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class AuditTier(str, Enum):
    TIER0 = "tier0"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


@dataclass(frozen=True)
class AuditDecision:
    tier: str
    risk_score: float
    should_run_heavy_probes: bool
    should_retain_full_telemetry: bool
    outcome: str
    reasons: list[str]


def compute_entropy(probs: Iterable[float]) -> float:
    import math

    vals = [max(1e-12, float(p)) for p in probs if p > 0]
    return float(-sum(p * math.log(p) for p in vals))


def compute_lightweight_risk(*, entropy: float, refusal_mass: float, affirmation_mass: float, factuality_score: float = 0.0, sycophancy_score: float = 0.0) -> float:
    risk = 0.0
    risk += min(1.0, entropy / 8.0) * 0.25
    risk += min(1.0, refusal_mass) * 0.15
    risk += min(1.0, affirmation_mass) * 0.10
    risk += min(1.0, factuality_score) * 0.25
    risk += min(1.0, sycophancy_score) * 0.25
    return float(min(1.0, risk))


def decide_audit_tier(*, audit_mode: str, risk_score: float, entropy: float, refusal_mass: float, has_high_risk_concept: bool, high_risk_suite: bool, entropy_threshold: float, refusal_mass_threshold: float, risk_threshold_review: float, risk_threshold_fail: float, selective_retention: bool) -> AuditDecision:
    reasons: list[str] = []
    if audit_mode == "off":
        return AuditDecision(AuditTier.TIER0.value, risk_score, False, False, "PASS", ["audit_off"])
    if audit_mode == "full" or high_risk_suite:
        reasons.append("tier3_full_audit")
        outcome = "FAIL" if risk_score >= risk_threshold_fail else ("REVIEW" if risk_score >= risk_threshold_review else "PASS")
        return AuditDecision(AuditTier.TIER3.value, risk_score, True, True, outcome, reasons)

    tier = AuditTier.TIER1.value
    heavy = False
    if audit_mode == "tiered" and (
        entropy >= entropy_threshold
        or refusal_mass >= refusal_mass_threshold
        or has_high_risk_concept
        or risk_score >= risk_threshold_review
    ):
        tier = AuditTier.TIER2.value
        heavy = True
        reasons.append("tier2_escalation")

    if entropy >= entropy_threshold:
        reasons.append("entropy_tripwire")
    if refusal_mass >= refusal_mass_threshold:
        reasons.append("refusal_mass_divergence")
    if has_high_risk_concept:
        reasons.append("high_risk_concept")

    outcome = "FAIL" if risk_score >= risk_threshold_fail else ("REVIEW" if risk_score >= risk_threshold_review else "PASS")
    retain = outcome in {"REVIEW", "FAIL", "FALLBACK"} or not selective_retention
    return AuditDecision(tier, risk_score, heavy or audit_mode == "cheap", retain, outcome, reasons)
