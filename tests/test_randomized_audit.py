import json
from dualstream.randomized_audit import randomized_selection

def test_same_nonce_same_selection():
    assert randomized_selection(99) == randomized_selection(99)

def test_different_nonce_changes_selection():
    assert randomized_selection(99) != randomized_selection(100)

def test_metadata_contains_hash_and_path_id():
    out = randomized_selection(77)
    assert out['audit_nonce_hash'] and out['audit_path_id']

def test_metadata_json_serializable():
    out = randomized_selection(77)
    json.dumps(out)
