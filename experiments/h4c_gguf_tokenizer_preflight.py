"""Check GGUF tokenizer suitability for H4-C development marker placement.

This preflight does no text generation, captures no protected audit key, and
never modifies the frozen PyTorch development scenarios. It verifies that the
existing exact valid-output placement grammar is expressible as stable single
generated-token words under the quantized model's llama.cpp tokenizer.

Passing is necessary, not sufficient: actual completions still need a separate
baseline competence gate and token-level position validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.h4c_gguf_c0_preflight import (
    _base_url,
    _check_server_model_path,
    _hash_model_file,
    _request,
)
from experiments.h4c_local_model_development import (
    SCHEMA_VERSION,
    _sha256_file,
    load_development_scenarios,
)


def _tokenize(endpoint: str, text: str, request_json) -> tuple[int, ...]:
    response = request_json(
        endpoint + "/tokenize",
        payload={
            "content": text,
            "add_special": False,
            "parse_special": False,
        },
    )
    ids = response.get("tokens")
    if not isinstance(ids, list) or any(
        type(token_id) is not int or token_id < 0 for token_id in ids
    ):
        raise ValueError("llama.cpp /tokenize returned malformed token IDs")
    return tuple(ids)


def inspect_scenario_tokenization(scenario, *, endpoint: str, request_json) -> dict:
    if (
        scenario.required_marker is None
        or scenario.filler_token is None
        or scenario.max_prefix_fillers is None
    ):
        raise ValueError("development scenario must define an exact placement grammar")

    success = scenario.success_marker
    filler = scenario.filler_token
    marker = scenario.required_marker
    success_ids = _tokenize(endpoint, success, request_json)
    filler_ids = _tokenize(endpoint, " " + filler, request_json)
    marker_ids = _tokenize(endpoint, " " + marker, request_json)
    one_filler = _tokenize(
        endpoint, f"{success} {filler} {marker}", request_json
    )
    two_fillers = _tokenize(
        endpoint, f"{success} {filler} {filler} {marker}", request_json
    )

    stable = (
        one_filler == success_ids + filler_ids + marker_ids
        and two_fillers == success_ids + filler_ids + filler_ids + marker_ids
    )
    single_token = (
        len(success_ids) == 1
        and len(filler_ids) == 1
        and len(marker_ids) == 1
    )
    return {
        "success_token_ids": list(success_ids),
        "filler_token_ids": list(filler_ids),
        "monitored_token_ids": list(marker_ids),
        "success_width": len(success_ids),
        "filler_width": len(filler_ids),
        "monitored_width": len(marker_ids),
        "canonical_sequence_stable": stable,
        "tokenization_compatible": single_token and stable,
    }


def run_gguf_tokenizer_preflight(
    *,
    server_url: str,
    gguf_path: Path,
    model_id: str,
    scenarios,
    scenario_set_sha256: str,
    request_json=_request,
) -> dict:
    if not model_id:
        raise ValueError("model_id must not be empty")
    if not scenarios:
        raise ValueError("no scenarios selected for GGUF tokenization")
    endpoint = _base_url(server_url)
    path = gguf_path.resolve(strict=True)
    file_hash = _hash_model_file(path)
    health = request_json(endpoint + "/health")
    if health.get("status") != "ok":
        raise ValueError("local llama.cpp server is not ready")
    props = request_json(endpoint + "/props")
    _check_server_model_path(props, path)

    results: dict[str, dict] = {}
    families: dict[str, str] = {}
    for scenario, marker, family in scenarios:
        if scenario.scenario_id in results:
            raise ValueError("duplicate development scenario ID")
        if marker != scenario.required_marker:
            raise ValueError("monitored marker differs from required marker")
        results[scenario.scenario_id] = inspect_scenario_tokenization(
            scenario, endpoint=endpoint, request_json=request_json
        )
        families[scenario.scenario_id] = family
    compatible = all(row["tokenization_compatible"] for row in results.values())
    return {
        "scope": (
            "Non-confirmatory GGUF H4-C development tokenization preflight only; "
            "no model generation, exposure estimate or confirmatory claim."
        ),
        "scenario_schema": SCHEMA_VERSION,
        "scenario_set_sha256": scenario_set_sha256,
        "inference_backend": "llama.cpp/native-tokenizer",
        "model_id": model_id,
        "gguf_filename": path.name,
        "gguf_sha256": file_hash,
        "model_revision": file_hash,
        "server_model_path": props["model_path"],
        "placement_tokenization_requirement": "single-token-stable-v1",
        "placement_tokenization": results,
        "scenario_families": families,
        "all_scenarios_tokenization_compatible": compatible,
        "checked_scenario_count": len(results),
        "incompatible_scenarios": [
            scenario_id for scenario_id, row in results.items()
            if not row["tokenization_compatible"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate GGUF tokenization of H4-C placement grammar; no inference"
    )
    parser.add_argument("--gguf-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--scenarios", type=Path,
        default=Path(__file__).with_name("h4c_model_development_scenarios.json"),
    )
    parser.add_argument(
        "--scenario-id", default="marker-placement-001",
        help="one development scenario ID, or 'all' (12 cases)",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--fail-on-incompatible", action="store_true")
    args = parser.parse_args()

    try:
        fixtures = load_development_scenarios(args.scenarios)
        if args.scenario_id != "all":
            fixtures = tuple(
                row for row in fixtures
                if row[0].scenario_id == args.scenario_id
            )
        report = run_gguf_tokenizer_preflight(
            server_url=args.server_url, gguf_path=args.gguf_path,
            model_id=args.model_id, scenarios=fixtures,
            scenario_set_sha256=_sha256_file(args.scenarios),
        )
    except (ValueError, OSError, FileNotFoundError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    if args.fail_on_incompatible and not report["all_scenarios_tokenization_compatible"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
