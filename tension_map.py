from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class TensionRule:
    rule_id: str
    selector_type: str
    selector_value: str
    widening_action: str
    expiry: float | None

    def applies_to(self, context: dict[str, Any]) -> bool:
        if self.expiry is not None and time.time() > self.expiry:
            return False
        
        # Simple string matching for now, expand based on selector_type
        if self.selector_type == "prompt_template_hash":
            return context.get("prompt_template_hash") == self.selector_value
        elif self.selector_type == "benchmark_family_id":
            return context.get("benchmark_family_id") == self.selector_value
        elif self.selector_type == "ast_signal":
            # context["ast_signals"] could be a list of AST codes like [301, 303]
            try:
                code = int(self.selector_value)
                return code in context.get("ast_signals", [])
            except ValueError:
                return False
        elif self.selector_type == "always":
            return True
        return False


@dataclass(frozen=True)
class TensionMap:
    map_id: int
    content_hash: bytes
    rules: list[TensionRule]

    @classmethod
    def parse_and_verify(cls, yaml_content: str | bytes, tension_keys: dict[str, bytes] | None = None) -> TensionMap:
        if isinstance(yaml_content, bytes):
            yaml_content = yaml_content.decode("utf-8")
            
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            raise ValueError("invalid tension map: expected dictionary")
            
        map_id = data.get("map_id", 0)
        signature_meta = data.get("signature")
        
        # To verify signature, we sign the canonicalized YAML without the signature block
        # For simplicity, we just sign the map_id and rules in a stable way
        if tension_keys is not None and signature_meta:
            signer_id = signature_meta.get("signer_id", "")
            key = tension_keys.get(signer_id)
            if not key:
                raise ValueError("tension map signed by unknown key")
            
            # Reconstruct signed payload (simplified for demonstration)
            payload = f"{map_id}:" + ",".join(f"{r['rule_id']}={r['selector_type']}={r['selector_value']}" for r in data.get("rules", []))
            expected = hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature_meta.get("hash", "")):
                raise ValueError("tension map signature mismatch")
        elif tension_keys is not None:
            raise ValueError("tension map requires signature when keys are provided")
            
        rules = []
        for r in data.get("rules", []):
            rules.append(TensionRule(
                rule_id=str(r.get("rule_id", "")),
                selector_type=str(r.get("selector_type", "")),
                selector_value=str(r.get("selector_value", "")),
                widening_action=str(r.get("widening_action", "widen")),
                expiry=float(r["expiry"]) if "expiry" in r else None
            ))
            
        content_hash = hashlib.sha256(yaml_content.encode("utf-8")).digest()
        return cls(map_id=int(map_id), content_hash=content_hash, rules=rules)

    def evaluate_triggers(self, context: dict[str, Any]) -> bool:
        """Returns True if any rule triggers widening based on the given context."""
        for rule in self.rules:
            if rule.applies_to(context):
                return True
        return False
