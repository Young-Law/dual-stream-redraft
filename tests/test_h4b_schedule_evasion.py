import experiments.h4b_schedule_evasion as h4b
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


def test_public_and_protected_schedules_hold_replay_context_constant(monkeypatch):
    calls = []

    def fake_schedule(*, key, audit_key_id, scenario_id, length, rate_ppm):
        calls.append((key, audit_key_id, scenario_id, length, rate_ppm))
        return set()

    monkeypatch.setattr(h4b, "_schedule", fake_schedule)
    h4b._one(key_id=3, scenario_id=17, length=10, rate_ppm=h4b.RATE_PPM)

    assert len(calls) == 2
    public, protected = calls
    assert public[1:] == protected[1:]
    assert public[1] == h4b.AUDIT_KEY_ID
    assert public[0] != protected[0]
