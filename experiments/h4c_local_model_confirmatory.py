"""Fail-closed local-model runner for frozen H4-C confirmatory execution.

This module is intentionally separate from smoke and development runners. It will not
execute unless a design-freeze manifest validates every frozen input and the runtime
configuration matches the currently installed H4-C treatment contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict
from importlib import metadata
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
from scripts.h4c_freeze_manifest import (
    validate_frozen_manifest,
    validate_runtime_config,
)

SCENARIO_SCHEMA = "h4c-model-confirmatory-v1"
RUNTIME_SCHEMA = "h4c-confirmatory-runtime-v1"
PUBLIC_SCHEDULE_KEY = hashlib.sha256(
    b"H4C-PUBLIC-CONFIRMATORY-SCHEDULE-v1"
).digest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_confirmatory_scenarios(
    path: Path,
) -> tuple[tuple[ScenarioFixture, str, str], ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCENARIO_SCHEMA:
        raise ValueError("unsupported H4-C confirmatory scenario schema")
    scope = str(data.get("scope", ""))
    if "confirmatory" not in scope.casefold():
        raise ValueError("confirmatory scenario scope must be explicit")
    rows = data.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("H4-C confirmatory scenarios must be a nonempty list")

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
        filler_token = str(row["filler_token"]).strip()
        max_prefix_fillers = int(row["max_prefix_fillers"])
        token_capacity = int(row["token_capacity"])

        if not family:
            raise ValueError(f"{scenario_id}: scenario_family must be nonempty")
        if not task_prompt or not success_marker:
            raise ValueError(f"{scenario_id}: prompt and success marker must be nonempty")
        if not monitored_marker or monitored_marker != required_marker:
            raise ValueError(
                f"{scenario_id}: monitored_marker must equal required_marker"
            )
        if not filler_token:
            raise ValueError(f"{scenario_id}: filler_token must be nonempty")
        if len({success_marker, monitored_marker, filler_token}) != 3:
            raise ValueError(
                f"{scenario_id}: success, monitored, and filler symbols must differ"
            )
        if max_prefix_fillers < 0:
            raise ValueError(f"{scenario_id}: max_prefix_fillers must be nonnegative")
        if token_capacity < max_prefix_fillers + 2:
            raise ValueError(
                f"{scenario_id}: token_capacity cannot represent every legal placement"
            )

        fixture = ScenarioFixture(
            scenario_id=scenario_id,
            task_prompt=task_prompt,
            success_marker=success_marker,
            token_capacity=token_capacity,
            dry_run_text=f"{success_marker} {required_marker}",
            dry_run_monitored_event_indices=(),
            required_marker=required_marker,
            filler_token=filler_token,
            max_prefix_fillers=max_prefix_fillers,
        )
        out.append((fixture, monitored_marker, family))
    return tuple(out)


def _load_runtime_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("confirmatory runtime config must be a JSON object")
    validate_runtime_config(data)
    if data.get("treatment_version") != TREATMENT_VERSION:
        raise ValueError("runtime treatment_version does not match installed treatment")
    return data


def _validate_environment(runtime: dict) -> None:
    expected = runtime["environment"]
    observed_python = platform.python_version()
    if observed_python != expected["python_version"]:
        raise ValueError(
            "frozen Python version mismatch: "
            f"expected {expected['python_version']}, observed {observed_python}"
        )
    observed_system = platform.system()
    if observed_system != expected["platform_system"]:
        raise ValueError(
            "frozen platform system mismatch: "
            f"expected {expected['platform_system']}, observed {observed_system}"
        )
    observed_machine = platform.machine()
    if observed_machine != expected["platform_machine"]:
        raise ValueError(
            "frozen platform machine mismatch: "
            f"expected {expected['platform_machine']}, observed {observed_machine}"
        )
    for package, expected_version in expected["packages"].items():
        try:
            observed_version = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise ValueError(f"frozen package is not installed: {package}") from exc
        if observed_version != expected_version:
            raise ValueError(
                f"frozen package version mismatch for {package}: "
                f"expected {expected_version}, observed {observed_version}"
            )


def _placement_tokenization_profile(
    adapter: LocalHFAdapter,
    scenarios: tuple[tuple[ScenarioFixture, str, str], ...],
) -> dict[str, dict[str, object]]:
    tokenizer = adapter.generator.tokenizer
    profile: dict[str, dict[str, object]] = {}
    failures: list[str] = []

    def encode(text: str) -> tuple[int, ...]:
        return tuple(
            int(x)
            for x in tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    for scenario, marker, _family in scenarios:
        if scenario.filler_token is None or scenario.max_prefix_fillers is None:
            failures.append(f"{scenario.scenario_id}: missing placement contract")
            continue
        success_ids = encode(scenario.success_marker)
        filler_ids = encode(" " + scenario.filler_token)
        monitored_ids = encode(" " + marker)
        one = encode(
            f"{scenario.success_marker} {scenario.filler_token} {marker}"
        )
        two = encode(
            f"{scenario.success_marker} {scenario.filler_token} "
            f"{scenario.filler_token} {marker}"
        )
        expected_one = success_ids + filler_ids + monitored_ids
        expected_two = success_ids + filler_ids + filler_ids + monitored_ids
        stable = one == expected_one and two == expected_two
        row = {
            "success_token_ids": list(success_ids),
            "filler_token_ids": list(filler_ids),
            "monitored_token_ids": list(monitored_ids),
            "success_width": len(success_ids),
            "filler_width": len(filler_ids),
            "monitored_width": len(monitored_ids),
            "canonical_sequence_stable": stable,
        }
        profile[scenario.scenario_id] = row
        if (
            len(success_ids) != 1
            or len(filler_ids) != 1
            or len(monitored_ids) != 1
            or not stable
        ):
            failures.append(
                f"{scenario.scenario_id}: placement symbols are not one-token stable"
            )
    if failures:
        raise ValueError("; ".join(failures))
    return profile


def _specs(
    scenario_id: str,
    *,
    model_id: str,
    model_revision: str,
    seed: int,
    runtime: dict,
) -> tuple[TrialSpec, TrialSpec, TrialSpec]:
    common = dict(
        scenario_id=scenario_id,
        model_id=model_id,
        model_revision=model_revision,
        seed=seed,
        audit_key_id=int(runtime["audit_key_id"]),
        rate_ppm=int(runtime["audit_rate_ppm"]),
        policy_version=str(runtime["policy_version"]),
        benchmark_id=str(runtime["benchmark_id"]),
        profile_id=str(runtime["profile_id"]),
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
        "task_prompt": request.task_prompt,
        "treatment_instruction": request.treatment_instruction,
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


def run_confirmatory(
    *,
    manifest_path: Path,
    allow_network: bool,
    cache_dir: str | None,
) -> dict:
    manifest = validate_frozen_manifest(manifest_path)
    files = manifest["files"]
    scenario_path = Path(files["scenario_set"]["path"])
    runtime_path = Path(files["runtime_config"]["path"])

    runtime = _load_runtime_config(runtime_path)
    _validate_environment(runtime)
    scenarios = load_confirmatory_scenarios(scenario_path)
    seeds = tuple(int(seed) for seed in runtime["seeds"])
    expected_pairs = len(scenarios) * len(seeds)
    if expected_pairs != manifest["number_of_confirmatory_pairs"]:
        raise ValueError("frozen pair count does not match scenarios x seeds")

    protected_key = _protected_key_from_env()
    marker_map = {
        scenario.scenario_id: marker
        for scenario, marker, _family in scenarios
    }
    decoding = runtime["decoding"]
    adapter = LocalHFAdapter(
        model_id=manifest["model_id"],
        model_revision=manifest["model_revision"],
        monitored_markers=marker_map,
        max_new_tokens=int(decoding["max_new_tokens"]),
        top_k=int(decoding["top_k"]),
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
        do_sample=bool(decoding["do_sample"]),
        repetition_penalty=float(decoding["repetition_penalty"]),
        no_repeat_ngram_size=int(decoding["no_repeat_ngram_size"]),
        local_files_only=not allow_network,
        device=runtime.get("device"),
        cache_dir=cache_dir,
    )
    placement_tokenization = _placement_tokenization_profile(adapter, scenarios)

    records: list[dict] = []
    for scenario, _marker, family in scenarios:
        for seed in seeds:
            c0, c1, c2 = _specs(
                scenario.scenario_id,
                model_id=manifest["model_id"],
                model_revision=manifest["model_revision"],
                seed=seed,
                runtime=runtime,
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

    return {
        "schema_version": "h4c-confirmatory-result-v1",
        "benchmark_id": runtime["benchmark_id"],
        "scope": (
            "Frozen H4-C confirmatory execution. Do not tune prompts, scenarios, "
            "runtime parameters, evaluator, or analysis from these outcomes."
        ),
        "design_freeze_manifest_sha256": _sha256_file(manifest_path),
        "frozen_git_commit": manifest["git_commit"],
        "scenario_set_sha256": files["scenario_set"]["sha256"],
        "runtime_config_sha256": files["runtime_config"]["sha256"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "resolved_model_revision": adapter.resolved_revision,
        "treatment_version": TREATMENT_VERSION,
        "monitoring_position_method": "generated-token-prefix-decode-v2",
        "prompt_format": adapter.prompt_format,
        "placement_tokenization_requirement": "single-token-stable-v1",
        "placement_tokenization": placement_tokenization,
        "runtime": runtime,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen H4-C confirmatory local-model trials"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = run_confirmatory(
            manifest_path=args.manifest,
            allow_network=args.allow_network,
            cache_dir=args.cache_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
