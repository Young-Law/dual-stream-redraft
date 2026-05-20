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


def test_falsy_manifests_do_not_collapse_and_wrong_manifest_fails():
    vals=[None,{},[],'',0,False]
    hashes=[randomized_selection(9, manifest_data=v)['randomized_manifest_hash'] for v in vals]
    assert len(set(hashes))==len(vals)
    sel = randomized_selection(9, manifest_data=False)
    assert not verify_randomized_selection(sel, nonce=9, manifest_data=0)


def test_none_domain_does_not_collide_with_user_manifest_payload():
    user_payload = ['__manifest_domain__', 'none']
    none_sel = randomized_selection(11, manifest_data=None)
    user_sel = randomized_selection(11, manifest_data=user_payload)
    assert none_sel['randomized_manifest_hash'] != user_sel['randomized_manifest_hash']
    assert not verify_randomized_selection(none_sel, nonce=11, manifest_data=user_payload)
