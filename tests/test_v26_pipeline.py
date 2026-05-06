from dualstream.audit_scheduler import compute_entropy, decide_audit_tier
from dualstream.randomized_audit import randomized_selection
from dualstream.fallback import FallbackRouter
from dualstream.frame import MonologueFrameV1, TopKToken
from dualstream.audit import coherence_outcome


def test_scheduler_escalates_entropy():
    d = decide_audit_tier(audit_mode='tiered', risk_score=0.2, entropy=4.5, entropy_threshold=4.0, refusal_mass=0.0, refusal_mass_threshold=0.2, high_risk_prompt=False, selective_retention=True)
    assert d.tier in {'tier2','tier3'}

def test_randomized_stable():
    a=randomized_selection(7); b=randomized_selection(7); c=randomized_selection(8)
    assert a==b and a!=c and a['audit_nonce_hash']

def test_fallback_no_loop():
    r=FallbackRouter(max_retries=1)
    d=r.route(retry_count=1, reason='FAIL', unchanged_retry_attempted=True)
    assert d.exhausted and d.action in {'canned_refusal','safe_override','abort'}

def test_frame_to_dict_has_v26_fields():
    fr=MonologueFrameV1(prompt_nonce=1,token_index=0,chosen_id=1,topk=[TopKToken(1,0.8)],audit_tier='tier1',audit_path_id='p',audit_nonce_hash='h')
    js=fr.to_dict()
    assert 'audit_tier' in js and 'audit_path_id' in js and 'audit_nonce_hash' in js

def test_coherence_factuality_affirmation_fail():
    fr=MonologueFrameV1(prompt_nonce=1,token_index=0,chosen_id=1,topk=[TopKToken(1,0.5)],audit_tier='tier2')
    from dualstream.frame import ConceptScore
    fr.concepts=[ConceptScore(2001,0.8)]
    out=coherence_outcome('yes that is true',[fr])
    assert out.outcome in {'FAIL','REVIEW'}
