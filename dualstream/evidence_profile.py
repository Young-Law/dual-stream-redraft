from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceProfileId(str, Enum):
    CI_LITE = "DSA-CI-Lite"
    CI_STANDARD = "DSA-CI-Standard"
    DEEP = "DSA-Deep"
    FORENSIC = "DSA-Forensic"


@dataclass(frozen=True)
class EvidenceProfile:
    profile_id: EvidenceProfileId
    ceiling_bytes_per_token: int
    base_k: int
    max_adaptive_k: int
    ci_modes: tuple[str, ...]
    verifier_budget_id: str
    verifier_time_seconds: float
    verifier_peak_mib: int
    minimum_budget_token_count: int = 10000
    adaptive_record_fraction_limit: float | None = None
    verifier_peak_rss_mib: int | None = None
    verifier_traced_peak_mib: int | None = None
    absolute_ci_safety_rail_mib: int | None = None
    default_pr_ci: bool = False
    adaptive_policy_id: str = "hybrid-rank-keyed-history-canary-v1"
    verifier_work_profile_id: str = "portable-work-v1"
    runtime_calibration_id: str = "runtime-calibration-v1"
    retention_policy_id: str = "local-floor-async-receipt-v1"
    default_stochastic_rate_ppm: int = 5000

    def __post_init__(self):
        if self.verifier_peak_rss_mib is None:
            object.__setattr__(self, "verifier_peak_rss_mib", self.verifier_peak_mib)

    def minimum_reconstructable_bytes_per_token(self, effective_topk: int | None = None) -> int:
        k = self.base_k if effective_topk is None else int(effective_topk)
        return 6 + max(0, k) * 5


PROFILES: dict[str, EvidenceProfile] = {
    EvidenceProfileId.CI_LITE.value: EvidenceProfile(EvidenceProfileId.CI_LITE, 24, 3, 10, ("pr", "normal", "nightly"), "h3e-ci-lite", 3.0, 512, 10000, 0.05, 512, 512, 1024, True),
    EvidenceProfileId.CI_STANDARD.value: EvidenceProfile(EvidenceProfileId.CI_STANDARD, 48, 5, 10, ("nightly", "release", "release-blocking"), "h3e-ci-standard", 6.0, 1024, 10000, 0.20, 1024, 1024, 2048),
    EvidenceProfileId.DEEP.value: EvidenceProfile(EvidenceProfileId.DEEP, 96, 5, 20, ("nightly", "probe", "adversarial"), "h3e-deep", 30.0, 2048, 10000, None, 2048, 2048, 4096),
    EvidenceProfileId.FORENSIC.value: EvidenceProfile(EvidenceProfileId.FORENSIC, 256, 5, 32, ("incident", "forensic", "replay"), "h3e-forensic", 120.0, 4096, 10000, None, 4096, 4096, 8192),
}


def get_evidence_profile(profile: str | EvidenceProfileId | EvidenceProfile) -> EvidenceProfile:
    if isinstance(profile, EvidenceProfile):
        return profile
    key = profile.value if isinstance(profile, EvidenceProfileId) else str(profile)
    try:
        return PROFILES[key]
    except KeyError as exc:
        raise ValueError(f"unknown evidence profile: {key}") from exc


def assert_profile_ci_mode(profile: str | EvidenceProfile, ci_mode: str) -> EvidenceProfile:
    prof = get_evidence_profile(profile)
    if ci_mode not in prof.ci_modes:
        raise ValueError(f"profile {prof.profile_id.value} is not compatible with ci-mode {ci_mode}")
    if ci_mode in {"pr", "normal"} and prof.profile_id is EvidenceProfileId.FORENSIC:
        raise ValueError("DSA-Forensic is not allowed as a default PR CI profile")
    return prof
