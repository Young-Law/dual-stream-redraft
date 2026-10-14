"""Non-confirmatory H4-C marker-placement development via local GGUF llama.cpp.

Reuses the existing development fixtures, treatment instructions, and audit
scheduler. Unlike the original PyTorch development runner, uses quantized GGUF
and native llama.cpp token IDs. All legal placement outputs must pass a
full-grammar tokenizer preflight before generation. Task-successful outputs
must use the canonical tokens exactly (with an optional single terminal EOS),
otherwise we fail closed rather than infer unsupported marker positions.

These data are never a substitute for the confirmatory H4-C measurement or
DSA pre-control top-k evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.h4c_gguf_smoke import GGUFSmokeAdapter
from experiments.h4c_gguf_tokenizer_preflight import (
    _tokenize,
    inspect_scenario_tokenization,
)
from experiments.h4c_local_model_development import (
    SCHEMA_VERSION,
    _parse_seeds,
    _sha256_file,
    load_development_scenarios,
)
from experiments.h4c_local_model_smoke import (
    PUBLIC_SCHEDULE_KEY,
    _protected_key_from_env,
    _run_one,
    _specs,
)
from experiments.h4c_model_in_loop import (
    TREATMENT_VERSION,
    assert_paired_public_context,
    legal_marker_indices,
    legal_placement_outputs,
)


def _strict_placement_profile(adapter, scenario) -> dict:
    """Verify the entire legal grammar (including zero and maximum fillers)."""
    profile = inspect_scenario_tokenization(
        scenario, endpoint=adapter.endpoint, request_json=adapter.request_json
    )
    if not profile["tokenization_compatible"]:
        raise ValueError(
            f"{scenario.scenario_id}: GGUF one-token placement preflight failed"
        )
    success_id = profile["success_token_ids"][0]
    filler_id = profile["filler_token_ids"][0]
    marker_id = profile["monitored_token_ids"][0]
    canonical: dict[int, list[int]] = {}
    for marker_index, output in legal_placement_outputs(scenario):
        expected = (success_id,) + (filler_id,) * (marker_index - 1) + (marker_id,)
        observed = _tokenize(adapter.endpoint, output, adapter.request_json)
        if observed != expected:
            raise ValueError(
                f"{scenario.scenario_id}: legal marker position {marker_index} "
                "does not use canonical single-token grammar"
            )
        canonical[marker_index] = list(expected)
    if set(canonical) != set(legal_marker_indices(scenario)):
        raise ValueError("legal placement preflight omitted a position")
    return {
        **profile,
        "all_legal_outputs_canonical": True,
        "legal_output_count": len(canonical),
    }


def _verify_task_successful_tokens(record: dict, scenario, profile: dict) -> None:
    """Fail closed if a valid-looking answer is not a canonical token sequence."""
    if record["result"]["task_success"] is not True:
        return
    positions = record["response"]["monitored_event_indices"]
    if len(positions) != 1 or positions[0] not in legal_marker_indices(scenario):
        raise ValueError(
            f"{scenario.scenario_id}: task success lacks one legal generated-token marker position"
        )
    observed = record["response"]["generated_token_ids"]
    marker_index = positions[0]
    expected = (
        profile["success_token_ids"]
        + profile["filler_token_ids"] * (marker_index - 1)
        + profile["monitored_token_ids"]
    )
    stop = record["generation_metadata"]["stop_type"]
    exact = observed == expected
    with_eos = (
        stop == "eos"
        and len(observed) == len(expected) + 1
        and observed[:len(expected)] == expected
    )
    if not (exact or with_eos):
        raise ValueError(
            f"{scenario.scenario_id}: successful text does not match canonical "
            "generated token sequence; cannot assign experimental marker placement"
        )
    record["generation_metadata"]["canonical_generated_tokens_verified"] = True
    record["generation_metadata"]["terminal_eos_token_present"] = with_eos


def run_gguf_development(
    adapter,
    *,
    scenarios,
    seeds: tuple[int, ...],
    rate_ppm: int,
    protected_key: bytes,
    scenario_set_sha256: str,
) -> dict:
    """One adapter, one model load, and a complete C0/C1/C2 sequence per seed."""
    if not scenarios:
        raise ValueError("no H4-C development scenarios selected")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("development seeds must be nonempty and unique")
    if not 0 <= rate_ppm <= 1_000_000:
        raise ValueError("rate_ppm must be between 0 and 1000000")
    if len(protected_key) < 16 or protected_key == PUBLIC_SCHEDULE_KEY:
        raise ValueError("invalid protected audit key")

    profiles: dict[str, dict] = {}
    for scenario, marker, _family in scenarios:
        if scenario.scenario_id in profiles:
            raise ValueError("duplicate H4-C development scenario")
        if adapter.monitored_markers.get(scenario.scenario_id) != marker:
            raise ValueError("GGUF adapter monitored marker differs from fixture")
        profiles[scenario.scenario_id] = _strict_placement_profile(adapter, scenario)

    records: list[dict] = []
    for scenario, _marker, family in scenarios:
        profile = profiles[scenario.scenario_id]
        for seed in seeds:
            c0, c1, c2 = _specs(
                scenario.scenario_id,
                model_id=adapter.model_id,
                model_revision=adapter.model_revision,
                seed=seed,
                rate_ppm=rate_ppm,
            )
            assert_paired_public_context(c1, c2)
            for spec, key in (
                (c0, protected_key),
                (c1, PUBLIC_SCHEDULE_KEY),
                (c2, protected_key),
            ):
                row = _run_one(spec, scenario, schedule_key=key, adapter=adapter)
                row["generation_metadata"] = dict(adapter.last_generation_metadata)
                row["scenario_family"] = family
                row["legal_marker_indices"] = list(legal_marker_indices(scenario))
                _verify_task_successful_tokens(row, scenario, profile)
                records.append(row)

    return {
        "benchmark_id": "H4C-MODEL-IN-LOOP-v1",
        "scope": (
            "Non-confirmatory GGUF H4-C development split only; "
            "not a preregistered H4-C confirmatory result. "
            "No DSA pre-control top-k evidence captured."
        ),
        "scenario_schema": SCHEMA_VERSION,
        "scenario_set_sha256": scenario_set_sha256,
        "inference_backend": "llama.cpp/native-completion",
        "chat_template_backend": "llama.cpp/apply-template",
        "model_id": adapter.model_id,
        "model_revision": adapter.model_revision,
        "gguf_filename": adapter.gguf_path.name,
        "gguf_sha256": adapter.model_revision,
        "server_model_path": adapter.server_model_path,
        "monitoring_position_method": "generated-token-prefix-decode-v2",
        "placement_tokenization_requirement": "single-token-stable-v1",
        "placement_tokenization": profiles,
        "treatment_version": TREATMENT_VERSION,
        "decoding": dict(adapter.decoding_config),
        "seeds": list(seeds),
        "rate_ppm": rate_ppm,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run non-confirmatory H4-C GGUF marker-placement development"
    )
    parser.add_argument("--gguf-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--scenarios", type=Path,
        default=Path(__file__).with_name("h4c_model_development_scenarios.json"),
    )
    parser.add_argument(
        "--scenario-id", default="marker-placement-001",
        help="one scenario ID, or 'all'; start with one on low-RAM machines",
    )
    parser.add_argument("--seeds", default="11")
    parser.add_argument("--rate-ppm", type=int, default=500_000)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        seeds = _parse_seeds(args.seeds)
        scenarios = load_development_scenarios(args.scenarios)
        if args.scenario_id != "all":
            scenarios = tuple(
                row for row in scenarios
                if row[0].scenario_id == args.scenario_id
            )
        protected_key = _protected_key_from_env()
        adapter = GGUFSmokeAdapter(
            server_url=args.server_url,
            gguf_path=args.gguf_path,
            model_id=args.model_id,
            monitored_markers={
                scenario.scenario_id: marker
                for scenario, marker, _family in scenarios
            },
            max_new_tokens=args.max_new_tokens,
        )
        report = run_gguf_development(
            adapter,
            scenarios=scenarios,
            seeds=seeds,
            rate_ppm=args.rate_ppm,
            protected_key=protected_key,
            scenario_set_sha256=_sha256_file(args.scenarios),
        )
    except (ValueError, OSError, FileNotFoundError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
