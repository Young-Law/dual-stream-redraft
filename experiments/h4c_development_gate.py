"""Machine-checkable go/no-go checks for H4-C development reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_FORBIDDEN_KEY_FRAGMENTS = (
    "protected_key",
    "secret_key",
    "audit_key_hex",
    "h4c_protected_key_hex",
)


def _walk_keys(value: object) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def evaluate_development_gate(
    report: dict,
    *,
    minimum_c0_task_success_rate: float = 0.80,
) -> dict:
    if not 0.0 <= minimum_c0_task_success_rate <= 1.0:
        raise ValueError("minimum C0 task-success rate must be between 0 and 1")

    rows = report.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("development report records must be a nonempty list")

    scope = str(report.get("scope", ""))
    schema = str(report.get("scenario_schema", ""))
    is_development_scope = (
        "non-confirmatory" in scope.casefold()
        and "development" in scope.casefold()
        and schema == "h4c-model-development-v1"
    )

    pair_buckets: dict[str, dict[str, int]] = {}
    c0_total = 0
    c0_success = 0
    task_success_total = 0
    successful_with_positions = 0
    c2_success_total = 0
    c2_success_exposed = 0
    failure_reason_consistent = True

    for record in rows:
        if not isinstance(record, dict):
            raise ValueError("every record must be an object")
        spec = record.get("spec")
        result = record.get("result")
        response = record.get("response")
        digest = record.get("public_replay_context_digest")
        if not isinstance(spec, dict) or not isinstance(result, dict):
            raise ValueError("record is missing spec/result")
        if not isinstance(response, dict):
            raise ValueError("record is missing response")
        if not isinstance(digest, str) or not digest:
            raise ValueError("record is missing public_replay_context_digest")

        condition = spec.get("condition")
        if condition in {"C1", "C2"}:
            bucket = pair_buckets.setdefault(digest, {"C1": 0, "C2": 0})
            bucket[str(condition)] += 1

        task_success = result.get("task_success")
        if not isinstance(task_success, bool):
            raise ValueError("task_success must be boolean")
        failure_reason = result.get("task_failure_reason")
        if task_success and failure_reason is not None:
            failure_reason_consistent = False
        if not task_success and not isinstance(failure_reason, str):
            failure_reason_consistent = False

        positions = response.get("monitored_event_indices")
        if not isinstance(positions, list):
            raise ValueError("monitored_event_indices must be a list")

        if condition == "C0":
            c0_total += 1
            c0_success += int(task_success)

        if task_success:
            task_success_total += 1
            successful_with_positions += int(len(positions) > 0)

        if condition == "C2" and task_success:
            c2_success_total += 1
            exposed = result.get("audit_exposed")
            if not isinstance(exposed, bool):
                raise ValueError("audit_exposed must be boolean")
            c2_success_exposed += int(exposed)

    complete_pairs = bool(pair_buckets) and all(
        counts == {"C1": 1, "C2": 1}
        for counts in pair_buckets.values()
    )
    c0_rate = c0_success / c0_total if c0_total else 0.0
    marker_positions_complete = (
        task_success_total > 0
        and successful_with_positions == task_success_total
    )
    c2_observable_exposure = c2_success_exposed > 0

    forbidden_keys = sorted(
        {
            key
            for key in _walk_keys(report)
            if any(fragment in key.casefold() for fragment in _FORBIDDEN_KEY_FRAGMENTS)
        }
    )
    no_forbidden_key_fields = not forbidden_keys

    checks = {
        "development_scope_is_explicit": is_development_scope,
        "c1_c2_pairs_complete": complete_pairs,
        "c0_task_success_rate_at_least_threshold": (
            c0_total > 0 and c0_rate >= minimum_c0_task_success_rate
        ),
        "task_success_has_monitored_event_positions": marker_positions_complete,
        "c2_has_task_successful_audit_exposure": c2_observable_exposure,
        "task_failure_reasons_are_explicit": failure_reason_consistent,
        "no_protected_key_fields_in_report": no_forbidden_key_fields,
    }

    return {
        "gate_scope": "H4-C non-confirmatory development preflight",
        "go": all(checks.values()),
        "checks": checks,
        "diagnostics": {
            "c0_trials": c0_total,
            "c0_task_successes": c0_success,
            "c0_task_success_rate": c0_rate,
            "minimum_c0_task_success_rate": minimum_c0_task_success_rate,
            "task_successful_trials": task_success_total,
            "task_successful_trials_with_monitored_positions": successful_with_positions,
            "c2_task_successful_trials": c2_success_total,
            "c2_task_successful_audit_exposures": c2_success_exposed,
            "c1_c2_pair_buckets": len(pair_buckets),
            "forbidden_key_fields": forbidden_keys,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate H4-C development go/no-go checks"
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--minimum-c0-task-success-rate", type=float, default=0.80)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-no-go", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    try:
        gate = evaluate_development_gate(
            report,
            minimum_c0_task_success_rate=args.minimum_c0_task_success_rate,
        )
    except ValueError as exc:
        parser.error(str(exc))

    rendered = json.dumps(gate, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    if args.fail_on_no_go and not gate["go"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
