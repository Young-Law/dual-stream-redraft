from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional
import json

from .compact import STABLE_METADATA_FIELDS
from .vocab import AST_MODEL_FACING_JSON_DEPTH_VIOLATION, AST_NESTED_CONTEXT_SERIALIZATION_RISK

DEFAULT_MAX_CONTEXT_DEPTH = 3
VISIBLE_SCOPES = {"prompt", "model", "review", "reviewer", "evaluator"}
MACHINE_SCOPES = {"machine", "internal", "storage", "transport", "schema"}

@dataclass
class ContextLintReport:
    maximum_json_depth: int = 0
    maximum_yaml_depth: int = 0
    prompt_visible_json_depth_violations: int = 0
    evaluator_visible_json_depth_violations: int = 0
    flat_yaml_context_bundle_count: int = 0
    deep_schema_reference_count: int = 0
    nested_context_serialization_failure_count: int = 0
    findings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.findings


def logical_depth(value: Any) -> int:
    if isinstance(value, Mapping):
        if not value:
            return 1
        return 1 + max(logical_depth(v) for v in value.values())
    if isinstance(value, list):
        if not value:
            return 1
        return 1 + max(logical_depth(v) for v in value)
    return 0


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], path))
        return out
    if isinstance(value, list):
        out = {}
        for idx, item in enumerate(value):
            path = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(_flatten(item, path))
        return out
    return {prefix: value}


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in ":#{}[],&*?|-<>=!%@`\n\r\t") or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text


def serialize_flat_yaml_context_bundle(bundle: Mapping[str, Any], *, namespace: str = "context") -> str:
    """Serialize model/reviewer context as flat YAML with stable path-like keys."""
    flattened = _flatten(bundle, namespace)
    return "\n".join(f"{key}: {_yaml_scalar(value)}" for key, value in sorted(flattened.items())) + "\n"


def yaml_depth(yaml_text: str) -> int:
    max_depth = 0
    for raw in yaml_text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        # Path-like keys (for example context.tool.0.name) are a flat representation,
        # not YAML nesting. Only indentation contributes to YAML depth.
        max_depth = max(max_depth, indent // 2 + 1)
    return max_depth


def _has_inline_deep_schema(value: Any, *, schema_ref_available: bool, max_depth: int) -> bool:
    if not schema_ref_available:
        return False
    if isinstance(value, Mapping):
        schema_like = any(k in value for k in ("$schema", "type", "properties", "required", "oneOf", "anyOf", "allOf"))
        if schema_like and logical_depth(value) > max_depth:
            return True
        return any(_has_inline_deep_schema(v, schema_ref_available=schema_ref_available, max_depth=max_depth) for v in value.values())
    if isinstance(value, list):
        return any(_has_inline_deep_schema(v, schema_ref_available=schema_ref_available, max_depth=max_depth) for v in value)
    return False


def _stable_repeated_in_nested_context(value: Any, bound_once: Iterable[str], *, depth: int = 0) -> int:
    if not isinstance(value, Mapping):
        if isinstance(value, list):
            return sum(_stable_repeated_in_nested_context(v, bound_once, depth=depth + 1) for v in value)
        return 0
    bound = set(bound_once)
    count = 0
    if depth > 0:
        count += len((set(value.keys()) & STABLE_METADATA_FIELDS) & bound)
    for v in value.values():
        count += _stable_repeated_in_nested_context(v, bound, depth=depth + 1)
    return count


def lint_context_payloads(payloads: List[Dict[str, Any]], *, max_depth: int = DEFAULT_MAX_CONTEXT_DEPTH) -> ContextLintReport:
    report = ContextLintReport()
    for payload in payloads:
        scope = str(payload.get("visibility", "machine")).lower()
        fmt = str(payload.get("format", "json")).lower()
        content = payload.get("content")
        schema_ref_available = bool(payload.get("schema_hash") or payload.get("schema_artifact_id") or payload.get("schema_ref"))
        bound_once = payload.get("agentops_bound_fields") or []
        visible = scope in VISIBLE_SCOPES
        evaluator = scope == "evaluator"

        if fmt == "json":
            depth = logical_depth(content)
            report.maximum_json_depth = max(report.maximum_json_depth, depth)
            if visible and depth > max_depth:
                key = "evaluator_visible_json_depth_violations" if evaluator else "prompt_visible_json_depth_violations"
                setattr(report, key, getattr(report, key) + 1)
                report.nested_context_serialization_failure_count += 1
                report.findings.append({"ast_code": AST_MODEL_FACING_JSON_DEPTH_VIOLATION, "message": f"{scope}-visible JSON depth {depth} exceeds {max_depth}"})
            if visible and _has_inline_deep_schema(content, schema_ref_available=schema_ref_available, max_depth=max_depth):
                report.deep_schema_reference_count += 1
                report.nested_context_serialization_failure_count += 1
                report.findings.append({"ast_code": AST_NESTED_CONTEXT_SERIALIZATION_RISK, "message": "deep JSON Schema inlined despite available schema hash/reference"})
            repeated = _stable_repeated_in_nested_context(content, bound_once)
            if visible and repeated:
                report.nested_context_serialization_failure_count += repeated
                report.findings.append({"ast_code": AST_NESTED_CONTEXT_SERIALIZATION_RISK, "message": "stable AgentOps metadata repeated in nested context packet"})
        elif fmt in {"yaml", "yml"}:
            text = str(content)
            depth = yaml_depth(text)
            report.maximum_yaml_depth = max(report.maximum_yaml_depth, depth)
            if visible:
                report.flat_yaml_context_bundle_count += 1
                if depth > max_depth:
                    report.nested_context_serialization_failure_count += 1
                    report.findings.append({"ast_code": AST_NESTED_CONTEXT_SERIALIZATION_RISK, "message": f"visible YAML depth {depth} exceeds {max_depth}"})
    return report


def verify_context_serialization(payloads: List[Dict[str, Any]], *, max_depth: int = DEFAULT_MAX_CONTEXT_DEPTH) -> Dict[str, Any]:
    report = lint_context_payloads(payloads, max_depth=max_depth)
    return {"context_serialization_passed": report.passed, "findings": report.findings, "lint_report": report}
