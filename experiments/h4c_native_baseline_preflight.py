"""Native Hugging Face C0 baseline preflight for low-memory H4-C development.

This is deliberately NOT an H4-C treatment trial. It uses the exact C0 prompt
construction and pinned checkpoint from the local smoke test, but calls the
model's standard Hugging Face `.generate()` path instead of DSA's custom
per-token decoder. Run one scenario at a time on RAM-constrained machines.

Compare this report with a C0 record from `h4c_local_model_smoke` using the
same model, revision, scenario, and max_new_tokens. No protected key is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.h4c_local_hf_adapter import LocalHFAdapter
from experiments.h4c_local_model_smoke import _specs, load_scenarios
from experiments.h4c_model_in_loop import (
    ModelRequest,
    ModelResponse,
    ScenarioFixture,
    build_model_request,
    evaluate_response,
)


def build_c0_prompt(request: ModelRequest, scenario: ScenarioFixture) -> str:
    """Mirror LocalHFAdapter.generate's C0 prompt exactly; never add a marker hint."""
    if request.spec.condition != "C0":
        raise ValueError("native baseline preflight supports C0 only")
    if request.scenario_id != scenario.scenario_id:
        raise ValueError("request/scenario mismatch")
    return (
        f"{scenario.task_prompt}\n\n"
        f"{request.treatment_instruction}\n"
        "Return only the final task output."
    )


def run_native_c0(
    adapter: LocalHFAdapter,
    request: ModelRequest,
    scenario: ScenarioFixture,
    monitored_marker: str,
    *,
    max_new_tokens: int,
) -> dict:
    """Run one native greedy C0 completion using the already loaded CPU model."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompt = build_c0_prompt(request, scenario)
    rendered_prompt = adapter.generator._render_prompt(prompt)
    tokenizer = adapter.generator.tokenizer
    model = adapter.generator.model
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    encoded = {
        key: value.to(adapter.generator.device)
        for key, value in encoded.items()
    }
    prompt_len = int(encoded["input_ids"].shape[1])
    stop_ids = sorted(adapter.generator._get_stop_token_ids())

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=min(max_new_tokens, scenario.token_capacity),
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            **({"eos_token_id": stop_ids} if stop_ids else {}),
        )
    generated_ids = tuple(int(x) for x in output_ids[0, prompt_len:].tolist())
    answer_text = str(tokenizer.decode(generated_ids, skip_special_tokens=True))
    monitored_indices = adapter._marker_positions(
        generated_ids, monitored_marker, answer_text
    )
    response = ModelResponse(
        text=answer_text,
        token_count=len(generated_ids),
        monitored_event_indices=monitored_indices,
        invalid=not generated_ids,
        generated_token_ids=generated_ids,
    )
    result = evaluate_response(
        request.spec, scenario, response, audit_schedule=()
    )

    return {
        "scope": (
            "Non-confirmatory native-HF C0 preflight only; not a confirmatory "
            "H4-C measurement and not a C1/C2 audit-evasion treatment."
        ),
        "inference_backend": "transformers.generate.greedy",
        "model_id": adapter.model_id,
        "model_revision": adapter.model_revision,
        "resolved_model_revision": adapter.resolved_revision,
        "prompt_format": adapter.prompt_format,
        "raw_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
        "scenario_id": scenario.scenario_id,
        "condition": "C0",
        "max_new_tokens": min(max_new_tokens, scenario.token_capacity),
        "generated_token_ids": list(generated_ids),
        "response_text": answer_text,
        "monitored_event_indices": list(monitored_indices),
        "task_success": result.task_success,
        "task_failure_reason": result.task_failure_reason,
        "parsed_choice": result.parsed_choice,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare DSA's failed C0 smoke baseline with native HF greedy generation"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("h4c_model_smoke_scenarios.json"))
    parser.add_argument("--scenario-id", default="style-choice-001")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rate-ppm", type=int, default=500_000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not 0 <= args.rate_ppm <= 1_000_000:
        parser.error("--rate-ppm must be between 0 and 1000000")

    matches = [
        (scenario, monitored_marker)
        for scenario, monitored_marker in load_scenarios(args.scenarios)
        if scenario.scenario_id == args.scenario_id
    ]
    if len(matches) != 1:
        parser.error(f"unknown or duplicate scenario: {args.scenario_id}")
    scenario, monitored_marker = matches[0]

    spec = _specs(
        scenario.scenario_id,
        model_id=args.model,
        model_revision=args.model_revision,
        seed=args.seed,
        rate_ppm=args.rate_ppm,
    )[0]
    request = build_model_request(spec, scenario, audit_schedule=())

    adapter = LocalHFAdapter(
        model_id=args.model,
        model_revision=args.model_revision,
        monitored_markers={scenario.scenario_id: monitored_marker},
        local_files_only=not args.allow_network,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    report = run_native_c0(
        adapter, request, scenario, monitored_marker,
        max_new_tokens=args.max_new_tokens,
    )
    report["seed"] = args.seed
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    print(rendered, end="")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
