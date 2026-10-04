"""Non-confirmatory H4-C local-model development runner.

This runner is deliberately separate from the smoke and future confirmatory paths. It
uses a development-only marker-placement scenario split to calibrate whether an actual
model can complete the task, emit observable monitored evidence, and adapt marker
placement when the predictable schedule is disclosed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.h4c_local_hf_adapter import LocalHFAdapter
from experiments.h4c_local_model_smoke import (
    PUBLIC_SCHEDULE_KEY,
    _protected_key_from_env,
    _run_one,
    _specs,
)
from experiments.h4c_model_in_loop import ScenarioFixture, assert_paired_public_context


SCHEMA_VERSION = "h4c-model-development-v1"
DEFAULT_SEEDS = (11, 17, 23, 29)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development_scenarios(
    path: Path,
) -> tuple[tuple[ScenarioFixture, str, str], ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported H4-C development scenario schema")
    rows = data.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("H4-C development scenarios must be a nonempty list")

    out: list[tuple[ScenarioFixture, str, str]] = []
    seen: set[str] = set()
    for row in rows:
        scenario_id = str(row["scenario_id"]).strip()
        if not scenario_id:
            raise ValueError("scenario_id must be nonempty")
        if scenario_id in seen:
            raise ValueError(f"duplicate scenario_id: {scenario_id}")
        seen.add(scenario_id)

        family = str(row.get("scenario_family", "")).strip()
        task_prompt = str(row["task_prompt"]).strip()
        success_marker = str(row["success_marker"]).strip()
        monitored_marker = str(row["monitored_marker"]).strip()
        required_marker = str(row["required_marker"]).strip()
        token_capacity = int(row["token_capacity"])

        if not family:
            raise ValueError(f"{scenario_id}: scenario_family must be nonempty")
        if not task_prompt or not success_marker:
            raise ValueError(f"{scenario_id}: prompt and success marker must be nonempty")
        if not monitored_marker or not required_marker:
            raise ValueError(f"{scenario_id}: monitored and required markers must be nonempty")
        if monitored_marker != required_marker:
            raise ValueError(
                f"{scenario_id}: development monitored_marker must equal required_marker"
            )
        if success_marker == monitored_marker:
            raise ValueError(f"{scenario_id}: success and monitored markers must differ")
        if token_capacity < 1:
            raise ValueError(f"{scenario_id}: token_capacity must be positive")

        fixture = ScenarioFixture(
            scenario_id=scenario_id,
            task_prompt=task_prompt,
            success_marker=success_marker,
            token_capacity=token_capacity,
            dry_run_text=f"{success_marker} {required_marker}",
            dry_run_monitored_event_indices=(),
            required_marker=required_marker,
        )
        out.append((fixture, monitored_marker, family))
    return tuple(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run non-confirmatory H4-C local-model development trials"
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).with_name("h4c_model_development_scenarios.json"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated development seeds",
    )
    parser.add_argument("--rate-ppm", type=int, default=100_000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not 0 <= args.rate_ppm <= 1_000_000:
        parser.error("--rate-ppm must be between 0 and 1000000")
    try:
        seeds = _parse_seeds(args.seeds)
        protected_key = _protected_key_from_env()
        scenarios = load_development_scenarios(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))

    marker_map = {
        scenario.scenario_id: marker
        for scenario, marker, _family in scenarios
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

    records: list[dict] = []
    for scenario, _marker, family in scenarios:
        for seed in seeds:
            c0, c1, c2 = _specs(
                scenario.scenario_id,
                model_id=args.model,
                model_revision=args.model_revision,
                seed=seed,
                rate_ppm=args.rate_ppm,
            )
            assert_paired_public_context(c1, c2)
            for spec, schedule_key in (
                (c0, protected_key),
                (c1, PUBLIC_SCHEDULE_KEY),
                (c2, protected_key),
            ):
                record = _run_one(
                    spec,
                    scenario,
                    schedule_key=schedule_key,
                    adapter=adapter,
                )
                record["scenario_family"] = family
                records.append(record)

    report = {
        "benchmark_id": "H4C-MODEL-IN-LOOP-v1",
        "scope": (
            "Non-confirmatory development split only; never pool these records into "
            "the preregistered H4-C confirmatory estimate."
        ),
        "scenario_schema": SCHEMA_VERSION,
        "scenario_set_sha256": _sha256_file(args.scenarios),
        "model_id": args.model,
        "model_revision": args.model_revision,
        "resolved_model_revision": adapter.resolved_revision,
        "monitoring_position_method": "generated-token-prefix-decode-v2",
        "decoding": adapter.decoding_config,
        "seeds": list(seeds),
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
