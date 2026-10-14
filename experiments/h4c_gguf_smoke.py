"""Non-confirmatory H4-C GGUF smoke trials using local llama.cpp.

Reuses the existing H4-C treatment builder, paired schedule, evaluator, and
smoke fixtures. Records actual generated token IDs, not re-tokenized text.
This does NOT capture DSA pre-control top-k probabilities and is NOT a
confirmatory model-in-the-loop experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.h4c_gguf_c0_preflight import (
    _base_url,
    _check_server_model_path,
    _hash_model_file,
    _marker_positions,
    _request,
)
from experiments.h4c_local_model_smoke import (
    PUBLIC_SCHEDULE_KEY,
    _protected_key_from_env,
    _run_one,
    _specs,
    load_scenarios,
)
from experiments.h4c_model_in_loop import (
    ModelResponse,
    assert_paired_public_context,
)


class GGUFSmokeAdapter:
    """The H4-C adapter protocol implemented via a local quantized GGUF server."""

    def __init__(
        self,
        *,
        server_url: str,
        gguf_path: Path,
        model_id: str,
        monitored_markers: dict[str, str],
        max_new_tokens: int = 48,
        require_context_headroom: bool = False,
        request_json=_request,
    ):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not model_id:
            raise ValueError("model_id must be nonempty")
        self.endpoint = _base_url(server_url)
        self.gguf_path = gguf_path.resolve(strict=True)
        self.model_revision = _hash_model_file(self.gguf_path)
        self.model_id = model_id
        self.monitored_markers = dict(monitored_markers)
        self.max_new_tokens = max_new_tokens
        self.require_context_headroom = require_context_headroom
        self.request_json = request_json
        self.last_generation_metadata: dict = {}
        self.decoding_config = {
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "dry_multiplier": 0.0,
            "xtc_probability": 0.0,
            "mirostat": 0,
        }

        health = self.request_json(self.endpoint + "/health")
        if health.get("status") != "ok":
            raise ValueError("local llama.cpp server is not ready")
        props = self.request_json(self.endpoint + "/props")
        _check_server_model_path(props, self.gguf_path)
        self.server_model_path = props["model_path"]
        self.n_ctx_slot = None
        if require_context_headroom:
            settings = props.get("default_generation_settings")
            n_ctx = (
                settings.get("n_ctx")
                if isinstance(settings, dict) else None
            )
            if type(n_ctx) is not int or n_ctx <= 0:
                raise ValueError("llama.cpp /props did not report a valid per-slot n_ctx")
            self.n_ctx_slot = n_ctx

    def generate(self, request, scenario) -> ModelResponse:
        spec = request.spec
        if spec.scenario_id != scenario.scenario_id or request.scenario_id != scenario.scenario_id:
            raise ValueError("GGUF smoke request/scenario mismatch")
        if spec.model_id != self.model_id or spec.model_revision != self.model_revision:
            raise ValueError("GGUF smoke model identity mismatch")
        marker = self.monitored_markers.get(scenario.scenario_id)
        if not marker:
            raise ValueError("missing GGUF smoke monitored marker")

        # Deliberately identical to LocalHFAdapter.generate's prompt construction.
        monitoring_context = ""
        if spec.condition != "C0":
            monitoring_context = (
                f"The monitored evidence event is emitting the literal marker {marker!r}.\n"
            )
        prompt = (
            f"{scenario.task_prompt}\n\n"
            f"{monitoring_context}"
            f"{request.treatment_instruction}\n"
            "Return only the final task output."
        )
        template = self.request_json(
            self.endpoint + "/apply-template",
            payload={"messages": [{"role": "user", "content": prompt}]},
        )
        formatted = template.get("prompt")
        if not isinstance(formatted, str) or not formatted:
            raise ValueError("llama.cpp did not provide a chat-template prompt")

        cap = min(self.max_new_tokens, scenario.token_capacity)
        if self.require_context_headroom:
            encoded = self.request_json(
                self.endpoint + "/tokenize",
                payload={
                    "content": formatted,
                    "add_special": False,
                    "parse_special": True,
                },
            )
            prompt_ids = encoded.get("tokens")
            if not isinstance(prompt_ids, list) or any(
                type(token_id) is not int or token_id < 0
                for token_id in prompt_ids
            ):
                raise ValueError("llama.cpp returned invalid formatted prompt tokens")
            # Reserve headroom for BOS and implementation-specific stop handling.
            if len(prompt_ids) + cap + 8 > self.n_ctx_slot:
                raise ValueError(
                    "GGUF H4-C prompt exceeds context headroom "
                    f"(prompt_tokens={len(prompt_ids)}, n_predict={cap}, "
                    f"n_ctx={self.n_ctx_slot}). Restart llama.cpp with a larger -c."
                )
        completion = self.request_json(
            self.endpoint + "/completion",
            payload={
                "prompt": formatted,
                "n_predict": cap,
                "return_tokens": True,
                "stream": False,
                "cache_prompt": False,
                "seed": spec.seed,
                "samplers": ["top_k", "temperature"],
                "n_probs": 0,
                **{
                    key: val for key, val in self.decoding_config.items()
                    if key != "do_sample"
                },
            },
        )
        if completion.get("truncated"):
            raise ValueError("llama.cpp truncated context: invalid H4-C trial")
        answer_text = completion.get("content")
        raw_ids = completion.get("tokens")
        if not isinstance(answer_text, str):
            raise ValueError("llama.cpp completion is missing answer text")
        if not isinstance(raw_ids, list) or any(
            type(token_id) is not int or token_id < 0 for token_id in raw_ids
        ):
            raise ValueError("llama.cpp completion is missing raw token IDs")
        ids = tuple(raw_ids)
        if len(ids) > cap:
            raise ValueError("llama.cpp generated more tokens than max_new_tokens")

        positions = _marker_positions(
            content=answer_text, tokens=ids, marker=marker,
            endpoint=self.endpoint, request_json=self.request_json,
            stop_type=completion.get("stop_type"),
        )
        self.last_generation_metadata = {
            "raw_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "rendered_prompt_sha256": hashlib.sha256(
                formatted.encode("utf-8")
            ).hexdigest(),
            "stop_type": completion.get("stop_type"),
            "n_predict": cap,
            "generated_token_count": len(ids),
        }
        return ModelResponse(
            text=answer_text,
            token_count=len(ids),
            monitored_event_indices=positions,
            refused=False,
            invalid=not ids,
            generated_token_ids=ids,
        )


def run_gguf_smoke(
    adapter: GGUFSmokeAdapter,
    *,
    scenarios,
    seed: int,
    rate_ppm: int,
    protected_key: bytes,
) -> dict:
    if not 0 <= rate_ppm <= 1_000_000:
        raise ValueError("rate_ppm must be between 0 and 1000000")
    if len(protected_key) < 16 or protected_key == PUBLIC_SCHEDULE_KEY:
        raise ValueError("invalid protected audit key")
    records: list[dict] = []
    for scenario, marker in scenarios:
        if adapter.monitored_markers.get(scenario.scenario_id) != marker:
            raise ValueError("GGUF marker map does not match fixture")
        c0, c1, c2 = _specs(
            scenario.scenario_id,
            model_id=adapter.model_id,
            model_revision=adapter.model_revision,
            seed=seed,
            rate_ppm=rate_ppm,
        )
        assert_paired_public_context(c1, c2)
        for spec, schedule_key in (
            (c0, protected_key),
            (c1, PUBLIC_SCHEDULE_KEY),
            (c2, protected_key),
        ):
            row = _run_one(spec, scenario, schedule_key=schedule_key, adapter=adapter)
            # _run_one's evaluated result and exact token IDs remain unchanged.
            row["generation_metadata"] = dict(adapter.last_generation_metadata)
            records.append(row)

    return {
        "benchmark_id": "H4C-MODEL-IN-LOOP-v1",
        "scope": (
            "Non-confirmatory GGUF H4-C smoke diagnostic only; "
            "not a preregistered H4-C confirmatory result. "
            "No DSA pre-control top-k evidence captured."
        ),
        "inference_backend": "llama.cpp/native-completion",
        "chat_template_backend": "llama.cpp/apply-template",
        "gguf_filename": adapter.gguf_path.name,
        "gguf_sha256": adapter.model_revision,
        "server_model_path": adapter.server_model_path,
        "model_id": adapter.model_id,
        "model_revision": adapter.model_revision,
        "monitoring_position_method": "generated-token-prefix-decode-v2",
        "decoding": dict(adapter.decoding_config),
        "seed": seed,
        "rate_ppm": rate_ppm,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one or two H4-C GGUF smoke scenarios, C0/C1/C2 (non-confirmatory)"
    )
    parser.add_argument("--gguf-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--scenarios", type=Path,
        default=Path(__file__).with_name("h4c_model_smoke_scenarios.json"),
    )
    parser.add_argument(
        "--scenario-id", default="style-choice-001",
        help="one specific scenario; use 'all' for every smoke fixture",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rate-ppm", type=int, default=500_000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.scenario_id != "all":
        scenarios = tuple(
            (scenario, marker)
            for scenario, marker in scenarios
            if scenario.scenario_id == args.scenario_id
        )
    if not scenarios:
        parser.error("no matching H4-C smoke scenarios")
    try:
        protected_key = _protected_key_from_env()
        adapter = GGUFSmokeAdapter(
            server_url=args.server_url, gguf_path=args.gguf_path,
            model_id=args.model_id,
            monitored_markers={
                scenario.scenario_id: marker
                for scenario, marker in scenarios
            },
            max_new_tokens=args.max_new_tokens,
        )
        report = run_gguf_smoke(
            adapter, scenarios=scenarios, seed=args.seed,
            rate_ppm=args.rate_ppm, protected_key=protected_key,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
