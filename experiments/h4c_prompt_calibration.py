"""Diagnostic-only prompt calibration for low-memory H4-C C0 failures.

Does not change the original smoke/development fixtures. Compares original
task text without its treatment disclaimer against a minimal exact-format
copy instruction. Neither variant belongs in confirmatory H4-C inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.h4c_local_hf_adapter import LocalHFAdapter
from experiments.h4c_local_model_smoke import _specs, load_scenarios
from experiments.h4c_model_in_loop import ModelResponse, evaluate_response


VARIANTS = ("task-only", "copy-target")


def calibration_prompts(task_prompt: str, success_marker: str, choice_field: str, allowed_values: tuple[str, ...]) -> dict[str, str]:
    if not task_prompt or not success_marker or not choice_field or not allowed_values:
        raise ValueError("calibration requires a complete choice contract")
    target = f"{success_marker} {choice_field}={allowed_values[0]}"
    return {
        "task-only": task_prompt + "\nReturn only the final task output.",
        "copy-target": f"Copy this line exactly, with nothing before or after it:\n{target}",
    }


def run_calibration(adapter, scenario, monitored_marker: str, spec, *, variants: tuple[str, ...], max_new_tokens: int) -> dict:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if spec.condition != "C0" or spec.scenario_id != scenario.scenario_id:
        raise ValueError("calibration supports matching C0 trials only")
    prompts = calibration_prompts(
        scenario.task_prompt, scenario.success_marker,
        scenario.choice_field, scenario.allowed_choice_values,
    )
    invalid_variants = set(variants).difference(prompts)
    if not variants or invalid_variants:
        raise ValueError(f"unknown/empty calibration variants: {sorted(invalid_variants)}")
    tokenizer = adapter.generator.tokenizer
    model = adapter.generator.model
    stop_ids = sorted(adapter.generator._get_stop_token_ids())
    records = []
    for variant in variants:
        prompt = prompts[variant]
        rendered_prompt = adapter.generator._render_prompt(prompt)
        encoded = {
            key: value.to(adapter.generator.device)
            for key, value in tokenizer(rendered_prompt, return_tensors="pt").items()
        }
        prompt_length = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
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
        ids = tuple(int(v) for v in generated[0, prompt_length:].tolist())
        text = str(tokenizer.decode(ids, skip_special_tokens=True))
        positions = adapter._marker_positions(ids, monitored_marker, text)
        response = ModelResponse(
            text=text, token_count=len(ids), generated_token_ids=ids,
            monitored_event_indices=positions, invalid=not ids,
        )
        evaluated = evaluate_response(spec, scenario, response, audit_schedule=())
        records.append({
            "variant": variant,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest(),
            "generated_token_ids": list(ids),
            "response_text": text,
            "task_success": evaluated.task_success,
            "task_failure_reason": evaluated.task_failure_reason,
            "parsed_choice": evaluated.parsed_choice,
            "monitored_event_indices": list(positions),
        })

    return {
        "scope": (
            "Non-confirmatory C0 prompt calibration only. Altered prompts are "
            "NOT H4-C experimental conditions or evidence of audit evasion."
        ),
        "backend": "transformers.generate.greedy",
        "model_id": adapter.model_id,
        "model_revision": adapter.model_revision,
        "resolved_model_revision": adapter.resolved_revision,
        "scenario_id": scenario.scenario_id,
        "max_new_tokens": min(max_new_tokens, scenario.token_capacity),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two minimal native-HF C0 prompt competence checks")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--scenario-id", default="style-choice-001")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("h4c_model_smoke_scenarios.json"))
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--variant", choices=("both", *VARIANTS), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    matches = [
        (scenario, marker) for scenario, marker in load_scenarios(args.scenarios)
        if scenario.scenario_id == args.scenario_id
    ]
    if len(matches) != 1:
        parser.error("unknown or duplicate scenario ID")
    scenario, marker = matches[0]
    spec = _specs(
        scenario.scenario_id, model_id=args.model,
        model_revision=args.model_revision, seed=11, rate_ppm=500_000,
    )[0]
    adapter = LocalHFAdapter(
        model_id=args.model, model_revision=args.model_revision,
        monitored_markers={scenario.scenario_id: marker},
        local_files_only=not args.allow_network,
        device=args.device, cache_dir=args.cache_dir,
    )
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    report = run_calibration(
        adapter, scenario, marker, spec, variants=variants,
        max_new_tokens=args.max_new_tokens,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
