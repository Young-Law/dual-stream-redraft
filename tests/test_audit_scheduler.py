from dualstream.audit_scheduler import compute_entropy, compute_mass_for_token_set, compute_lightweight_risk, decide_audit_tier, AuditSchedulerConfig, LightweightRiskFeatures
from dualstream.vocab import *


def test_entropy_helpers_edge_cases():
    assert compute_entropy([]) == 0.0
    assert compute_entropy([1.0]) == 0.0
    assert compute_entropy([2.0, 2.0]) > 0
    assert compute_entropy([1000.0, 1001.0], input_is_logits=True) > 0


def test_mass_helpers():
    assert compute_mass_for_token_set([], {1}) == 0.0
    assert compute_mass_for_token_set([("no", 0.2), ("yes", 0.8)], {"yes"}) > 0.7
    assert compute_mass_for_token_set([0.1, 0.9], {1}) > 0.8


def test_ci_retention_rules():
    for mode in ("nightly", "release-blocking", "deep"):
        d = decide_audit_tier(audit_mode="tiered", risk_score=0.1, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True, ci_mode=mode)
        assert d.tier == "tier3" and d.should_run_heavy_probes and d.should_retain_full_telemetry and d.retention_decision == "FULL"
    d2 = decide_audit_tier(audit_mode="tiered", risk_score=0.1, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True, ci_mode="targeted")
    assert d2.retention_decision == "COMPACT"


def test_ast_behaviors_and_outcomes():
    review = decide_audit_tier(audit_mode="tiered", risk_score=0.2, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True, ast_scores={AST_PREMISE_LIKELY_FALSE:0.5})
    assert review.outcome in {"REVIEW", "PASS"}
    fallback = decide_audit_tier(audit_mode="tiered", risk_score=0.1, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True, ast_scores={AST_FALLBACK_ROUTE_INVOKED:0.9})
    assert fallback.outcome == "FALLBACK"
    fail = decide_audit_tier(audit_mode="tiered", risk_score=0.1, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True, ast_scores={AST_RETRY_BUDGET_EXCEEDED:1.0})
    assert fail.outcome == "FAIL"


def test_structural_fail_escalates_tier3():
    d = decide_audit_tier(audit_mode="tiered", risk_score=0.1, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True, missing_frames=True)
    assert d.tier == "tier3" and d.outcome == "FAIL"


def test_compute_lightweight_risk_clamped():
    cfg = AuditSchedulerConfig()
    r, reasons = compute_lightweight_risk(LightweightRiskFeatures(entropy=9.0, refusal_mass=0.9, structural_errors=3, high_risk_prompt=True), cfg)
    assert 0.0 <= r <= 1.0
    assert reasons
