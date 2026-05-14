from dualstream.randomized_audit import randomized_selection, verify_randomized_selection

def test_deterministic_by_nonce_manifest():
    a = randomized_selection(42, manifest_data={'a':1})
    b = randomized_selection(42, manifest_data={'a':1})
    c = randomized_selection(43, manifest_data={'a':1})
    assert a == b
    assert a != c

def test_verify_tamper_fails():
    sel = randomized_selection(7, manifest_data={'x':2})
    assert verify_randomized_selection(sel, nonce=7, manifest_data={'x':2})
    bad = dict(sel); bad['audit_path_id'] = 'path-x'
    assert not verify_randomized_selection(bad, nonce=7, manifest_data={'x':2})
