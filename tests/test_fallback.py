from dualstream.fallback import FallbackRouter, DEFAULT_FALLBACK

def test_routes_on_fail_state():
    r = FallbackRouter(max_retries=1)
    d = r.route(retry_count=1, reason='FAIL')
    assert d.exhausted and d.action == 'canned_refusal'

def test_retry_budget_enforced():
    r = FallbackRouter(max_retries=2)
    assert r.route(retry_count=1, reason='FAIL').action == 'retry'
    assert r.route(retry_count=2, reason='FAIL').exhausted

def test_canned_refusal_deterministic():
    r = FallbackRouter(max_retries=0)
    a = r.route(retry_count=0, reason='FAIL')
    b = r.route(retry_count=0, reason='FAIL')
    assert a.fallback_text == b.fallback_text == DEFAULT_FALLBACK

def test_no_infinite_retry_behavior():
    r = FallbackRouter(max_retries=1)
    d = r.route(retry_count=0, reason='FAIL', unchanged_retry_attempted=True)
    assert d.exhausted
