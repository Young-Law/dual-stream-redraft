from dualstream.audit_scheduler import decide_audit_tier, compute_entropy, compute_lightweight_risk

def test_low_risk_fast_pass():
    d = decide_audit_tier(audit_mode='tiered', risk_score=0.1, entropy=1.0, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True)
    assert d.tier == 'tier1' and d.outcome == 'PASS'

def test_high_entropy_escalates():
    d = decide_audit_tier(audit_mode='tiered', risk_score=0.2, entropy=5.0, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True)
    assert d.tier == 'tier2'

def test_high_concept_risk_escalates():
    risk = compute_lightweight_risk(entropy=1.0, refusal_mass=0.0, concept_scores={3101: 0.8})
    d = decide_audit_tier(audit_mode='tiered', risk_score=risk, entropy=1.0, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True)
    assert d.outcome in {'REVIEW', 'FAIL'}

def test_full_mode_selects_full_audit():
    d = decide_audit_tier(audit_mode='full', risk_score=0.1, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True)
    assert d.tier == 'tier3' and d.should_run_heavy_probes


def test_deep_ci_forces_retention_even_on_pass():
    from dualstream.audit_scheduler import decide_audit_tier
    d = decide_audit_tier(audit_mode='tiered', risk_score=0.1, entropy=0.1, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True, ci_mode='nightly')
    assert d.should_run_heavy_probes is True
    assert d.should_retain_full_telemetry is True
    assert d.metrics['retention_enabled'] is True
