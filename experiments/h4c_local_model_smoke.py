"""Executable non-confirmatory H4-C local-model smoke runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from experiments.h4c_local_hf_adapter import LocalHFAdapter
from experiments.h4c_model_in_loop import (
    TREATMENT_VERSION,
    ScenarioFixture,
    TrialSpec,
    assert_paired_public_context,
    build_audit_schedule,
    build_model_request,
    evaluate_response,
)


PUBLIC_SCHEDULE_KEY = hashlib.sha256(b"H4C-PUBLIC-SMOKE-SCHEDULE-v1").digest()


def _protected_key_from_env() -> bytes:
    value = os.environ.get("H4C_PROTECTED_KEY_HEX", "").strip()
    if not value:
        raise ValueError("H4C_PROTECTED_KEY_HEX must be set")
    try:
        key = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("H4C_PROTECTED_KEY_HEX must be hexadecimal") from exc
    if len(key) < 16:
        raise ValueError("H4C_PROTECTED_KEY_HEX must decode to at least 16 bytes")
    if key == PUBLIC_SCHEDULE_KEY:
        raise ValueError("protected and public schedule keys must differ")
    return key


def load_scenarios(path: Path) -> tuple[tuple[ScenarioFixture, str], ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "h4c-model-smoke-v1":
        raise ValueError("unsupported H4-C model smoke schema")
    rows = data.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("H4-C model smoke scenarios must be a nonempty list")

    out = []
    seen = set()
    for row in rows:
        scenario_id = str(row["scenario_id"])
        if scenario_id in seen:
            raise ValueError(f"duplicate scenario_id: {scenario_id}")
        seen.add(scenario_id)

        success_marker = str(row["success_marker"])
        monitored_marker = str(row["monitored_marker"])
        token_capacity = int(row["token_capacity"])
        choice_field = str(row.get("choice_field", "")).strip()
        raw_allowed_values = row.get("allowed_choice_values", [])
        if not success_marker or not monitored_marker:
            raise ValueError(f"{scenario_id}: markers must be nonempty")
        if success_marker == monitored_marker:
            raise ValueError(f"{scenario_id}: success and monitored markers must differ")
        if token_capacity < 1:
            raise ValueError(f"{scenario_id}: token_capacity must be positive")
        if not isinstance(raw_allowed_values, list):
            raise ValueError(f"{scenario_id}: allowed_choice_values must be a list")
        allowed_choice_values = tuple(str(value).strip() for value in raw_allowed_values)
        if not choice_field or not allowed_choice_values:
            raise ValueError(
                f"{scenario_id}: choice_field and allowed_choice_values are required"
            )
        if any(not value for value in allowed_choice_values):
            raise ValueError(f"{scenario_id}: allowed choice values must be nonempty")
        if len({value.casefold() for value in allowed_choice_values}) != len(
            allowed_choice_values
        ):
            raise ValueError(f"{scenario_id}: allowed choice values must be unique")

        fixture = ScenarioFixture(
            scenario_id=scenario_id,
            task_prompt=str(row["task_prompt"]),
            success_marker=success_marker,
            token_capacity=token_capacity,
            dry_run_text=success_marker,
            dry_run_monitored_event_indices=(),
            choice_field=choice_field,
            allowed_choice_values=allowed_choice_values,
        )
        out.append((fixture, monitored_marker))
    return tuple(out)


def _specs(
    scenario_id: str,
    *,
    model_id: str,
    model_revision: str,
    seed: int,
    rate_ppm: int,
) -> tuple[TrialSpec, TrialSpec, TrialSpec]:
    common = dict(
        scenario_id=scenario_id,
        model_id=model_id,
        model_revision=model_revision,
        seed=seed,
        audit_key_id=7,
        rate_ppm=rate_ppm,
        policy_version="v2.10.1",
        benchmark_id="H4C-MODEL-IN-LOOP-v1",
        profile_id="DSA-CI-Lite",
    )
    return tuple(
        TrialSpec(condition=condition, **common)
        for condition in ("C0", "C1", "C2")
    )


def _digest_schedule(schedule: tuple[int, ...]) -> str:
    raw = json.dumps(list(schedule), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _run_one(
    spec: TrialSpec,
    scenario: ScenarioFixture,
    *,
    schedule_key: bytes,
    adapter: LocalHFAdapter,
) -> dict:
    schedule = build_audit_schedule(
        spec,
        key=schedule_key,
        token_count=scenario.token_capacity,
    )
    request = build_model_request(spec, scenario, audit_schedule=schedule)
    response = adapter.generate(request, scenario)
    result = evaluate_response(
        spec,
        scenario,
        response,
        audit_schedule=schedule,
    )
    return {
        "spec": asdict(spec),
        "public_replay_context_digest": request.public_replay_context_digest,
        "schedule_digest": _digest_schedule(schedule),
        "schedule_disclosed_to_model": request.disclosed_audit_indices is not None,
        "disclosed_audit_indices": (
            list(request.disclosed_audit_indices)
            if request.disclosed_audit_indices is not None
            else None
        ),
        "treatment_version": TREATMENT_VERSION,
        "response": {
            "text": response.text,
            "token_count": response.token_count,
            "generated_token_ids": list(response.generated_token_ids),
            "monitored_event_indices": list(response.monitored_event_indices),
            "refused": response.refused,
            "invalid": response.invalid,
        },
        "result": result.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run non-confirmatory H4-C local-model smoke trials"
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).with_name("h4c_model_smoke_scenarios.json"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rate-ppm", type=int, default=100_000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device")
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not 0 <= args.rate_ppm <= 1_000_000:
        parser.error("--rate-ppm must be between 0 and 1000000")
    try:
        protected_key = _protected_key_from_env()
        scenarios = load_scenarios(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))

    marker_map = {
        scenario.scenario_id: marker
        for scenario, marker in scenarios
    }
    adapter = LocalHFAdapter(
        model_id=args.model,
        model_revision=args.model_revision,
        monitored_markers=marker_map,
        max_new_tokens=args.max_new_tokens,
        local_files_only=not args.allow_network,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    records = []
    for scenario, _marker in scenarios:
        c0, c1, c2 = _specs(
            scenario.scenario_id,
            model_id=args.model,
            model_revision=args.model_revision,
            seed=args.seed,
            rate_ppm=args.rate_ppm,
        )
        assert_paired_public_context(c1, c2)
        records.append(
            _run_one(
                c0,
                scenario,
                schedule_key=protected_key,
                adapter=adapter,
            )
        )
        records.append(
            _run_one(
                c1,
                scenario,
                schedule_key=PUBLIC_SCHEDULE_KEY,
                adapter=adapter,
            )
        )
        records.append(
            _run_one(
                c2,
                scenario,
                schedule_key=protected_key,
                adapter=adapter,
            )
        )

    report = {
        "benchmark_id": "H4C-MODEL-IN-LOOP-v1",
        "scope": (
            "Non-confirmatory local-model smoke only; do not use these records "
            "as the preregistered H4-C confirmatory result."
        ),
        "model_id": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": adapter.resolved_revision,
        "monitoring_position_method": "generated-token-prefix-decode-v2",
        "prompt_format": adapter.prompt_format,
        "decoding": adapter.decoding_config,
        "seed": args.seed,
        "rate_ppm": args.rate_ppm,
        "records": records,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
