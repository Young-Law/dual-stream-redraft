import time
import hashlib
import hmac

import pytest
import yaml

from dualstream.tension_map import TensionMap, TensionRule, canonical_tension_map_payload


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
    assert rule.applies_to({"benchmark_family_id": "arc-eval"}) is True
    assert rule.applies_to({"benchmark_family_id": "other-eval"}) is False

    rule_ast = TensionRule(
        rule_id="r2",
        selector_type="ast_signal",
        selector_value="303",
        widening_action="widen",
        expiry=None,
    )
    assert rule_ast.applies_to({"ast_signals": [301, 303]}) is True
    assert rule_ast.applies_to({"ast_signals": [301, 302]}) is False

    rule_expired = TensionRule(
        rule_id="r3",
        selector_type="always",
        selector_value="",
        widening_action="widen",
        expiry=time.time() - 10,
    )
    assert rule_expired.applies_to({}) is False

    rule_active = TensionRule(
        rule_id="r4",
        selector_type="always",
        selector_value="",
        widening_action="widen",
        expiry=time.time() + 10,
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


def _signed_map(key, signer_id="gov-authority", *, expiry=4102444800.0, widening_action="widen", extra=None):
    data = {
        "map_id": 200,
        "rules": [{
            "rule_id": "r1",
            "selector_type": "benchmark_family_id",
            "selector_value": "arc-eval",
            "widening_action": widening_action,
            "expiry": expiry,
        }],
    }
    if extra:
        data.update(extra)
    signature_hash = hmac.new(
        key, canonical_tension_map_payload(data), hashlib.sha256
    ).hexdigest()
    data["signature"] = {"signer_id": signer_id, "hash": signature_hash}
    return yaml.safe_dump(data, sort_keys=False)


def test_signed_tension_map_governance():
    key = b"tension-map-symmetric-signing-key"
    signer_id = "gov-authority"
    yaml_content = _signed_map(key, signer_id)

    tmap = TensionMap.parse_and_verify(yaml_content, tension_keys={signer_id: key})
    assert tmap.map_id == 200

    with pytest.raises(ValueError, match="unknown key"):
        TensionMap.parse_and_verify(yaml_content, tension_keys={"other-signer": key})

    unsigned = yaml.safe_load(yaml_content)
    unsigned.pop("signature")
    with pytest.raises(ValueError, match="requires signature"):
        TensionMap.parse_and_verify(
            yaml.safe_dump(unsigned, sort_keys=False), tension_keys={signer_id: key}
        )

    tampered = yaml.safe_load(yaml_content)
    tampered["signature"]["hash"] = "wronghashabcdef"
    with pytest.raises(ValueError, match="signature mismatch"):
        TensionMap.parse_and_verify(
            yaml.safe_dump(tampered, sort_keys=False), tension_keys={signer_id: key}
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda data: data["rules"][0].__setitem__("expiry", 7258118400.0),
        lambda data: data["rules"][0].pop("expiry"),
        lambda data: data["rules"][0].__setitem__("widening_action", "disabled"),
        lambda data: data.__setitem__("future_policy_field", "changed"),
    ],
)
def test_signed_tension_map_rejects_policy_field_tampering(mutator):
    key = b"tension-map-symmetric-signing-key"
    signer_id = "gov-authority"
    data = yaml.safe_load(_signed_map(key, signer_id, extra={"future_policy_field": "original"}))
    mutator(data)

    with pytest.raises(ValueError, match="signature mismatch"):
        TensionMap.parse_and_verify(
            yaml.safe_dump(data, sort_keys=False), tension_keys={signer_id: key}
        )


def test_signed_tension_map_expiry_controls_behavior_only_when_authenticated():
    key = b"tension-map-symmetric-signing-key"
    signer_id = "gov-authority"
    expired = _signed_map(key, signer_id, expiry=1.0)
    tmap = TensionMap.parse_and_verify(expired, tension_keys={signer_id: key})
    assert tmap.evaluate_triggers({"benchmark_family_id": "arc-eval"}) is False

    data = yaml.safe_load(expired)
    data["rules"][0]["expiry"] = 7258118400.0
    with pytest.raises(ValueError, match="signature mismatch"):
        TensionMap.parse_and_verify(
            yaml.safe_dump(data, sort_keys=False), tension_keys={signer_id: key}
        )
