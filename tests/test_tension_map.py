import time
import hashlib
import hmac
from dualstream.tension_map import TensionMap, TensionRule


def test_tension_map_parsing_without_signature():
    yaml_content = """
map_id: 100
rules:
  - rule_id: "r1"
    selector_type: "benchmark_family_id"
    selector_value: "arc-eval"
    widening_action: "widen"
  - rule_id: "r2"
    selector_type: "always"
    selector_value: ""
    widening_action: "widen"
"""
    tmap = TensionMap.parse_and_verify(yaml_content)
    assert tmap.map_id == 100
    assert len(tmap.rules) == 2
    assert tmap.rules[0].rule_id == "r1"
    assert tmap.rules[0].selector_type == "benchmark_family_id"
    assert tmap.rules[0].selector_value == "arc-eval"


def test_tension_rule_application():
    rule = TensionRule(
        rule_id="r1",
        selector_type="benchmark_family_id",
        selector_value="arc-eval",
        widening_action="widen",
        expiry=None,
    )
    # Applies to matching context
    assert rule.applies_to({"benchmark_family_id": "arc-eval"}) is True
    # Does not apply to mismatching context
    assert rule.applies_to({"benchmark_family_id": "other-eval"}) is False

    # Check ast_signal selector
    rule_ast = TensionRule(
        rule_id="r2",
        selector_type="ast_signal",
        selector_value="303",
        widening_action="widen",
        expiry=None,
    )
    assert rule_ast.applies_to({"ast_signals": [301, 303]}) is True
    assert rule_ast.applies_to({"ast_signals": [301, 302]}) is False

    # Check expiry
    rule_expired = TensionRule(
        rule_id="r3",
        selector_type="always",
        selector_value="",
        widening_action="widen",
        expiry=time.time() - 10,  # 10 seconds in the past
    )
    assert rule_expired.applies_to({}) is False

    rule_active = TensionRule(
        rule_id="r4",
        selector_type="always",
        selector_value="",
        widening_action="widen",
        expiry=time.time() + 10,  # 10 seconds in the future
    )
    assert rule_active.applies_to({}) is True


def test_tension_map_evaluation():
    yaml_content = """
map_id: 101
rules:
  - rule_id: "r1"
    selector_type: "benchmark_family_id"
    selector_value: "arc-eval"
    widening_action: "widen"
"""
    tmap = TensionMap.parse_and_verify(yaml_content)
    assert tmap.evaluate_triggers({"benchmark_family_id": "arc-eval"}) is True
    assert tmap.evaluate_triggers({"benchmark_family_id": "other-eval"}) is False


def test_signed_tension_map_governance():
    key = b"tension-map-symmetric-signing-key"
    signer_id = "gov-authority"
    map_id = 200
    
    # Payload reconstruction logic from TensionMap.parse_and_verify:
    # f"{map_id}:" + ",".join(f"{r['rule_id']}={r['selector_type']}={r['selector_value']}" for r in rules)
    payload = f"{map_id}:r1=benchmark_family_id=arc-eval"
    signature_hash = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    yaml_content = f"""
map_id: {map_id}
rules:
  - rule_id: "r1"
    selector_type: "benchmark_family_id"
    selector_value: "arc-eval"
    widening_action: "widen"
signature:
  signer_id: "{signer_id}"
  hash: "{signature_hash}"
"""
    # 1. Verification succeeds with valid key
    tmap = TensionMap.parse_and_verify(yaml_content, tension_keys={signer_id: key})
    assert tmap.map_id == map_id

    # 2. Verification raises ValueError with unknown signer_id
    try:
        TensionMap.parse_and_verify(yaml_content, tension_keys={"other-signer": key})
        assert False, "Expected ValueError for unknown signer_id"
    except ValueError as exc:
        assert "unknown key" in str(exc)

    # 3. Verification raises ValueError if signature block is missing but keys are expected
    yaml_unsigned = f"""
map_id: {map_id}
rules:
  - rule_id: "r1"
    selector_type: "benchmark_family_id"
    selector_value: "arc-eval"
    widening_action: "widen"
"""
    try:
        TensionMap.parse_and_verify(yaml_unsigned, tension_keys={signer_id: key})
        assert False, "Expected ValueError for missing signature block"
    except ValueError as exc:
        assert "requires signature" in str(exc)

    # 4. Verification raises ValueError with tampered signature hash
    yaml_tampered = f"""
map_id: {map_id}
rules:
  - rule_id: "r1"
    selector_type: "benchmark_family_id"
    selector_value: "arc-eval"
    widening_action: "widen"
signature:
  signer_id: "{signer_id}"
  hash: "wronghashabcdef"
"""
    try:
        TensionMap.parse_and_verify(yaml_tampered, tension_keys={signer_id: key})
        assert False, "Expected ValueError for tampered signature"
    except ValueError as exc:
        assert "signature mismatch" in str(exc)
