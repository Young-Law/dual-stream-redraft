from dualstream.randomized_audit import compute_manifest_hash, randomized_selection, verify_randomized_selection


def test_manifest_hash_distinguishes_falsy_values():
    assert compute_manifest_hash(None) != compute_manifest_hash({})
    assert compute_manifest_hash({}) != compute_manifest_hash([])
    assert compute_manifest_hash([]) != compute_manifest_hash("")
    assert compute_manifest_hash("") != compute_manifest_hash(0)
    assert compute_manifest_hash(0) != compute_manifest_hash(False)


def test_selection_and_verification_manifest_bound():
    a = randomized_selection(42, manifest_data=[])
    b = randomized_selection(42, manifest_data=[])
    c = randomized_selection(42, manifest_data={})
    assert a == b
    assert a["randomized_manifest_hash"] != c["randomized_manifest_hash"]
    assert verify_randomized_selection(a, nonce=42, manifest_data=[])
    assert not verify_randomized_selection(a, nonce=42, manifest_data={})


def test_tamper_fails():
    sel = randomized_selection(7, manifest_data={"x": 2})
    bad_sel = dict(sel)
    bad_sel["audit_path_id"] = "path-x"
    assert not verify_randomized_selection(bad_sel, nonce=7, manifest_data={"x": 2})
    assert not verify_randomized_selection(sel, nonce=7, manifest_data={"x": 3})
