"""Quantized GGUF C0 competency preflight using a local llama.cpp server.

This is intentionally separate from the PyTorch DSA generator and from H4-C's
confirmatory protocol. The native llama.cpp API supplies the model's own chat
template and generated token IDs. Never infer token positions by re-tokenizing
decoded output. The model is pinned by SHA-256 of the local GGUF file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from experiments.h4c_local_model_smoke import _specs, load_scenarios
from experiments.h4c_model_in_loop import (
    ModelResponse,
    build_model_request,
    evaluate_response,
    exact_marker_spans,
)


def _base_url(raw: str) -> str:
    parsed = urlparse(raw)
    if parsed.scheme != "http" or parsed.hostname not in {
        "localhost", "127.0.0.1", "::1",
    } or parsed.username or parsed.password or parsed.path not in ("", "/"):
        raise ValueError("server must be a loopback HTTP URL without credentials or path")
    return raw.rstrip("/")


def _request(url: str, *, payload: dict | None = None, timeout: int = 300) -> dict:
    """JSON-only standard-library client. No cloud API or Python LLM dependency."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as result:
        raw = result.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("llama.cpp response exceeds 4 MB")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("llama.cpp returned a non-object JSON response")
    return parsed


def _hash_model_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"GGUF model file not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_server_model_path(props: dict, model_path: Path) -> None:
    server_path = props.get("model_path")
    if not isinstance(server_path, str) or not server_path:
        raise ValueError("llama.cpp /props did not declare the model_path")
    candidate = Path(server_path)
    if not candidate.exists():
        raise ValueError(
            "Cannot resolve server model_path. Launch llama.cpp with an absolute "
            "GGUF filename in -m so its identity can be verified."
        )
    if not os.path.samefile(candidate, model_path):
        raise ValueError(
            "llama.cpp server loaded a different GGUF file than --gguf-path"
        )


def _prompt(request, scenario) -> str:
    if request.spec.condition != "C0":
        raise ValueError("GGUF preflight supports C0 only")
    if request.scenario_id != scenario.scenario_id:
        raise ValueError("request/scenario mismatch")
    return (
        f"{scenario.task_prompt}\n\n"
        f"{request.treatment_instruction}\n"
        "Return only the final task output."
    )


def _marker_positions(
    *,
    content: str,
    tokens: tuple[int, ...],
    marker: str,
    endpoint: str,
    request_json,
) -> tuple[int, ...]:
    spans = exact_marker_spans(content, marker)
    if not spans:
        return ()
    prefixes = [""]
    for end in range(1, len(tokens) + 1):
        response = request_json(
            endpoint + "/detokenize", payload={"tokens": list(tokens[:end])}
        )
        decoded = response.get("content")
        if not isinstance(decoded, str):
            raise ValueError("llama.cpp /detokenize returned no text")
        if not decoded.startswith(prefixes[-1]):
            raise ValueError("unstable prefix decoding; cannot claim marker positions")
        prefixes.append(decoded)
    if prefixes[-1] != content:
        raise ValueError("token detokenization differs from completion content")
    indices: set[int] = set()
    for start, stop in spans:
        for index in range(len(tokens)):
            if len(prefixes[index + 1]) > start and len(prefixes[index]) < stop:
                indices.add(index)
    return tuple(sorted(indices))


def run_gguf_c0(
    *,
    server_url: str,
    gguf_path: Path,
    model_id: str,
    scenario,
    monitored_marker: str,
    seed: int,
    rate_ppm: int,
    max_new_tokens: int,
    request_json=_request,
) -> dict:
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if not 0 <= rate_ppm <= 1_000_000:
        raise ValueError("--rate-ppm must be between 0 and 1000000")
    endpoint = _base_url(server_url)
    path = gguf_path.resolve(strict=True)
    # SHA-256 is computed locally, never inferred from a model's display name.
    revision = _hash_model_file(path)
    spec = _specs(
        scenario.scenario_id, model_id=model_id,
        model_revision=revision, seed=seed, rate_ppm=rate_ppm,
    )[0]
    request = build_model_request(spec, scenario, audit_schedule=())
    raw_prompt = _prompt(request, scenario)

    health = request_json(endpoint + "/health")
    if health.get("status") != "ok":
        raise ValueError("local llama.cpp server is not ready")
    props = request_json(endpoint + "/props")
    _check_server_model_path(props, path)

    template = request_json(
        endpoint + "/apply-template",
        payload={"messages": [{"role": "user", "content": raw_prompt}]},
    )
    formatted = template.get("prompt")
    if not isinstance(formatted, str) or not formatted:
        raise ValueError("llama.cpp failed to apply model chat template")

    limit = min(max_new_tokens, scenario.token_capacity)
    decoding = {
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
    completion = request_json(
        endpoint + "/completion",
        payload={
            "prompt": formatted,
            "n_predict": limit,
            "return_tokens": True,
            "stream": False,
            "cache_prompt": False,
            "seed": seed,
            "samplers": ["top_k", "temperature"],
            "n_probs": 0,
            **decoding,
        },
    )
    if completion.get("truncated"):
        raise ValueError("llama.cpp truncated the context; invalid diagnostic")
    content = completion.get("content")
    raw_tokens = completion.get("tokens")
    if not isinstance(content, str):
        raise ValueError("llama.cpp completion has no text")
    if not isinstance(raw_tokens, list) or any(
        type(token) is not int or token < 0 for token in raw_tokens
    ):
        raise ValueError("llama.cpp must return raw generated integer token IDs")
    tokens = tuple(raw_tokens)
    if len(tokens) > limit:
        raise ValueError("llama.cpp generated more tokens than requested")
    positions = _marker_positions(
        content=content, tokens=tokens, marker=monitored_marker,
        endpoint=endpoint, request_json=request_json,
    )
    response = ModelResponse(
        text=content, token_count=len(tokens),
        monitored_event_indices=positions, invalid=not tokens,
        generated_token_ids=tokens,
    )
    evaluation = evaluate_response(spec, scenario, response, audit_schedule=())
    return {
        "scope": (
            "Non-confirmatory local quantized GGUF C0 preflight only. "
            "No H4-C inference or DSA top-k evidence-provenance claim."
        ),
        "inference_backend": "llama.cpp/native-completion",
        "model_id": model_id,
        "model_revision": revision,
        "gguf_sha256": revision,
        "gguf_filename": path.name,
        "server_model_path": props["model_path"],
        "chat_template_backend": "llama.cpp/apply-template",
        "raw_prompt_sha256": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(formatted.encode("utf-8")).hexdigest(),
        "scenario_id": scenario.scenario_id,
        "condition": "C0",
        "seed": seed,
        "rate_ppm": rate_ppm,
        "max_new_tokens": limit,
        "decoding": decoding,
        "generated_token_ids": list(tokens),
        "response_text": content,
        "stop_type": completion.get("stop_type"),
        "monitored_event_indices": list(positions),
        "task_success": evaluation.task_success,
        "task_failure_reason": evaluation.task_failure_reason,
        "parsed_choice": evaluation.parsed_choice,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Native llama.cpp GGUF C0 competency test; never confirmatory"
    )
    parser.add_argument("--gguf-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--scenarios", type=Path,
        default=Path(__file__).with_name("h4c_model_smoke_scenarios.json"),
    )
    parser.add_argument("--scenario-id", default="style-choice-001")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rate-ppm", type=int, default=500_000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    matches = [
        (scenario, marker)
        for scenario, marker in load_scenarios(args.scenarios)
        if scenario.scenario_id == args.scenario_id
    ]
    if len(matches) != 1:
        parser.error("unknown or duplicated scenario ID")
    scenario, marker = matches[0]
    try:
        report = run_gguf_c0(
            server_url=args.server_url, gguf_path=args.gguf_path,
            model_id=args.model_id, scenario=scenario,
            monitored_marker=marker, seed=args.seed,
            rate_ppm=args.rate_ppm, max_new_tokens=args.max_new_tokens,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
