from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import yaml


def canonical_tension_map_payload(data: dict[str, Any]) -> bytes:
    """Return the canonical bytes authenticated by a tension-map signature.

    Every parsed map field is covered except the signature block itself. This
    deliberately authenticates behavior-affecting fields such as expiry and
    widening_action, and it makes subsequently added policy fields signed by
    default instead of requiring a hand-maintained allowlist.
    """
    if not isinstance(data, dict):
        raise ValueError("invalid tension map: expected dictionary")
    unsigned = {key: value for key, value in data.items() if key != "signature"}
    try:
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("tension map contains non-canonical policy values") from exc
    return canonical.encode("utf-8")


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

        # Authenticate the complete parsed policy document, excluding only the
        # signature container itself. Formatting and YAML comments do not affect
        # the signature, while every semantic policy field does.
        if tension_keys is not None and signature_meta:
            if not isinstance(signature_meta, dict):
                raise ValueError("invalid tension map signature metadata")
            signer_id = signature_meta.get("signer_id", "")
            key = tension_keys.get(signer_id)
            if not key:
                raise ValueError("tension map signed by unknown key")

            payload = canonical_tension_map_payload(data)
            expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, str(signature_meta.get("hash", ""))):
                raise ValueError("tension map signature mismatch")
        elif tension_keys is not None:
            raise ValueError("tension map requires signature when keys are provided")

        rules = []
        raw_rules = data.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ValueError("invalid tension map: rules must be a list")
        for r in raw_rules:
            if not isinstance(r, dict):
                raise ValueError("invalid tension map: each rule must be a dictionary")
            expiry = float(r["expiry"]) if "expiry" in r else None
            if expiry is not None and not math.isfinite(expiry):
                raise ValueError("invalid tension map: expiry must be finite")
            rules.append(TensionRule(
                rule_id=str(r.get("rule_id", "")),
                selector_type=str(r.get("selector_type", "")),
                selector_value=str(r.get("selector_value", "")),
                widening_action=str(r.get("widening_action", "widen")),
                expiry=expiry,
            ))

        content_hash = hashlib.sha256(yaml_content.encode("utf-8")).digest()
        return cls(map_id=int(map_id), content_hash=content_hash, rules=rules)

    def evaluate_triggers(self, context: dict[str, Any]) -> bool:
        """Returns True if any rule triggers widening based on the given context."""
        for rule in self.rules:
            if rule.applies_to(context):
                return True
        return False
