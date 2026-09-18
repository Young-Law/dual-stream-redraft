from experiments.h4b_schedule_evasion import analytical_touch_probability, run

def test_analytical_probability_is_monotone():
    values=[analytical_touch_probability(x) for x in (1,10,50,100,500)]
    assert values == sorted(values)
    assert 0 < values[0] < values[-1] < 1

def test_schedule_aware_attacker_avoids_public_schedule_in_small_fixture():
    report=run(keys=2, scenarios_per_key=2)
    results=report["results"]
    for length in ("1","10","50","100"):
        assert results[length]["schedule_aware_public"]["mean"] == 0.0

def test_scope_does_not_claim_semantic_detection():
    report=run(keys=1, scenarios_per_key=1)
    scope=report["scope"].lower()
    assert "does not establish deceptive-alignment detection" in scope
    assert "semantic validity" in scope
