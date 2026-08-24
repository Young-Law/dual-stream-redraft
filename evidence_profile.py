from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# V3.3 header field identifiers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V33HeaderFields:
    """Numeric identifiers for V3.3 binary header fields.

    Maps to the integer fields in the ``_HEADER_V33`` struct:
    ``tokenizer_id`` (uint16), ``signal_schema_id`` (uint16),
    ``quantization_id`` (uint8), ``verifier_work_profile_id`` (uint16),
    ``runtime_calibration_id`` (uint16), ``retention_policy_id`` (uint16).
    """
    tokenizer_id: int = 1
    signal_schema_id: int = 1
    quantization_id: int = 1
    verifier_work_profile_id: int = 1
    runtime_calibration_id: int = 1
    retention_policy_id: int = 1


# --- Well-known field-registry entries ---

# Tokenizer identifiers
TOKENIZER_ID_GENERIC = 1

# Signal schema identifiers
SIGNAL_SCHEMA_ID_AST1_V210 = 1

# Quantization identifiers
QUANTIZATION_ID_UINT8_PROB_V1 = 1

# Verifier work-profile identifiers
VERIFIER_WORK_PORTABLE_V1 = 1001
VERIFIER_WORK_BATCH_V1 = 1002
VERIFIER_WORK_FORENSIC_V1 = 1003

# Runtime calibration identifiers
RUNTIME_CALIBRATION_LOCAL_V1 = 2001
RUNTIME_CALIBRATION_ONLINE_V1 = 2002

# Retention policy identifiers
RETENTION_POLICY_LOCAL_FLOOR_V1 = 3001
RETENTION_POLICY_ASYNC_RECEIPT_V1 = 3002
RETENTION_POLICY_FORENSIC_HOLD_V1 = 3003

# --- Per-profile V3.3 header field bundles ---

CI_LITE_V33_FIELDS = V33HeaderFields(
    tokenizer_id=TOKENIZER_ID_GENERIC,
    signal_schema_id=SIGNAL_SCHEMA_ID_AST1_V210,
    quantization_id=QUANTIZATION_ID_UINT8_PROB_V1,
    verifier_work_profile_id=VERIFIER_WORK_PORTABLE_V1,
    runtime_calibration_id=RUNTIME_CALIBRATION_LOCAL_V1,
    retention_policy_id=RETENTION_POLICY_LOCAL_FLOOR_V1,
)

CI_STANDARD_V33_FIELDS = V33HeaderFields(
    tokenizer_id=TOKENIZER_ID_GENERIC,
    signal_schema_id=SIGNAL_SCHEMA_ID_AST1_V210,
    quantization_id=QUANTIZATION_ID_UINT8_PROB_V1,
    verifier_work_profile_id=VERIFIER_WORK_PORTABLE_V1,
    runtime_calibration_id=RUNTIME_CALIBRATION_LOCAL_V1,
    retention_policy_id=RETENTION_POLICY_ASYNC_RECEIPT_V1,
)

DEEP_V33_FIELDS = V33HeaderFields(
    tokenizer_id=TOKENIZER_ID_GENERIC,
    signal_schema_id=SIGNAL_SCHEMA_ID_AST1_V210,
    quantization_id=QUANTIZATION_ID_UINT8_PROB_V1,
    verifier_work_profile_id=VERIFIER_WORK_BATCH_V1,
    runtime_calibration_id=RUNTIME_CALIBRATION_ONLINE_V1,
    retention_policy_id=RETENTION_POLICY_ASYNC_RECEIPT_V1,
)

FORENSIC_V33_FIELDS = V33HeaderFields(
    tokenizer_id=TOKENIZER_ID_GENERIC,
    signal_schema_id=SIGNAL_SCHEMA_ID_AST1_V210,
    quantization_id=QUANTIZATION_ID_UINT8_PROB_V1,
    verifier_work_profile_id=VERIFIER_WORK_FORENSIC_V1,
    runtime_calibration_id=RUNTIME_CALIBRATION_ONLINE_V1,
    retention_policy_id=RETENTION_POLICY_FORENSIC_HOLD_V1,
)


# Valid ranges for decode-time validation
V33_HEADER_FIELD_RANGES: dict[str, tuple[int, int]] = {
    "tokenizer_id": (0, 0xFFFF),
    "signal_schema_id": (0, 0xFFFF),
    "quantization_id": (0, 0xFF),
    "verifier_work_profile_id": (0, 0xFFFF),
    "runtime_calibration_id": (0, 0xFFFF),
    "retention_policy_id": (0, 0xFFFF),
}


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
    v33_header_fields: V33HeaderFields = V33HeaderFields()

    def __post_init__(self):
        if self.verifier_peak_rss_mib is None:
            object.__setattr__(self, "verifier_peak_rss_mib", self.verifier_peak_mib)

    def minimum_reconstructable_bytes_per_token(self, effective_topk: int | None = None) -> int:
        k = self.base_k if effective_topk is None else int(effective_topk)
        return 6 + max(0, k) * 5


PROFILES: dict[str, EvidenceProfile] = {
    EvidenceProfileId.CI_LITE.value: EvidenceProfile(
        EvidenceProfileId.CI_LITE, 24, 3, 10, ("pr", "normal", "nightly"),
        "h3e-ci-lite", 15.0, 512, 10000, 0.05, 512, 512, 1024, True,
        v33_header_fields=CI_LITE_V33_FIELDS,
    ),
    EvidenceProfileId.CI_STANDARD.value: EvidenceProfile(
        EvidenceProfileId.CI_STANDARD, 48, 5, 10, ("nightly", "release", "release-blocking"),
        "h3e-ci-standard", 6.0, 1024, 10000, 0.20, 1024, 1024, 2048,
        v33_header_fields=CI_STANDARD_V33_FIELDS,
    ),
    EvidenceProfileId.DEEP.value: EvidenceProfile(
        EvidenceProfileId.DEEP, 96, 5, 20, ("nightly", "probe", "adversarial"),
        "h3e-deep", 30.0, 2048, 10000, None, 2048, 2048, 4096,
        v33_header_fields=DEEP_V33_FIELDS,
    ),
    EvidenceProfileId.FORENSIC.value: EvidenceProfile(
        EvidenceProfileId.FORENSIC, 256, 5, 32, ("incident", "forensic", "replay"),
        "h3e-forensic", 120.0, 4096, 10000, None, 4096, 4096, 8192,
        v33_header_fields=FORENSIC_V33_FIELDS,
    ),
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
