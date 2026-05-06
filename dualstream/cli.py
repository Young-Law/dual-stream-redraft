from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .audit import coherence_audit
from .arc_solver import ArcSolver, SolverConfig, write_submission, write_task_artifacts
from .arc_task import load_task, load_tasks_from_dir
from .render import render_monologue_text

if TYPE_CHECKING:
    from .generator import DualStreamGenerator, GenerationConfig

# Test-time monkeypatch hooks (keeps generate path behavior unchanged)
DualStreamGenerator: Any = None
GenerationConfig: Any = None


def _load_generator_types() -> tuple[type[Any], type[Any]]:
    global DualStreamGenerator, GenerationConfig
    if DualStreamGenerator is not None and GenerationConfig is not None:
        return DualStreamGenerator, GenerationConfig
    if DualStreamGenerator is not None and GenerationConfig is None:
        class _ShimGenerationConfig:
            def __init__(self, **kwargs: Any):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        GenerationConfig = _ShimGenerationConfig
        return DualStreamGenerator, _ShimGenerationConfig

    from .generator import DualStreamGenerator as _DualStreamGenerator, GenerationConfig as _GenerationConfig

    DualStreamGenerator = _DualStreamGenerator
    GenerationConfig = _GenerationConfig
    return _DualStreamGenerator, _GenerationConfig


def _load_prompt_file(path: str) -> list[str]:
    prompt_path = Path(path)

    if not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file is not a regular file: {prompt_path}")

    try:
        text = prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"Could not read prompt file {prompt_path}: {exc}") from exc

    prompts = [line for line in text.splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"No non-empty prompts found in: {prompt_path}")

    return prompts


def _run_generation(gen: Any, cfg: Any, prompt: str, outdir: Path) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    result = gen.generate(prompt, cfg)

    frames = result["frames"]
    monologue_text = render_monologue_text(frames, tokenizer_decode=gen.tokenizer.decode)

    answer_path = outdir / "answer.txt"
    monologue_path = outdir / "monologue.txt"
    monologue_jsonl_path = outdir / "monologue.jsonl"
    monologue_bin_path = outdir / "monologue.bin"
    meta_path = outdir / "meta.json"
    audit_path = outdir / "audit.json"

    answer_path.write_text(result["answer"], encoding="utf-8")
    monologue_path.write_text(monologue_text, encoding="utf-8")

    with monologue_jsonl_path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr.to_dict(), ensure_ascii=False) + "\n")

    with monologue_bin_path.open("wb") as f:
        for fb in result["frame_bytes"]:
            f.write(fb)

    meta = {
        "prompt_nonce": result["prompt_nonce"],
        "model": result["model"],
        "running_hash": result["running_hash"],
        "config": {
            "model": cfg.model,
            "max_new_tokens": cfg.max_new_tokens,
            "top_k": cfg.top_k,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "do_sample": cfg.do_sample,
            "include_attn": cfg.include_attn,
            "include_probes": cfg.include_probes,
            "probe_pack_path": cfg.probe_pack_path,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    findings = coherence_audit(
        result["answer"],
        frames,
        decode_token=lambda tid: gen.tokenizer.decode([tid]),
    )
    audit_path.write_text(
        json.dumps([f.__dict__ for f in findings], indent=2),
        encoding="utf-8",
    )

    return {
        "answer_path": answer_path.name,
        "monologue_path": monologue_path.name,
        "audit_path": audit_path.name,
        "meta_path": meta_path.name,
    }


def cmd_generate(args: argparse.Namespace) -> int:
    prompts: list[str] | None = None
    if args.prompt is None:
        prompts = _load_prompt_file(args.prompt_file)

    DualStreamGeneratorCls, GenerationConfigCls = _load_generator_types()

    cfg = GenerationConfigCls(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
        seed=args.seed,
        include_attn=args.include_attn,
        include_probes=args.include_probes,
        probe_pack_path=args.probe_pack,
        enable_heuristics=not args.no_heuristics,
        include_crc32=not args.no_crc32,
        include_running_hash=not args.no_running_hash,
        device=args.device,
        local_files_only=args.offline,
        cache_dir=args.cache_dir,
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gen = DualStreamGeneratorCls(
        cfg.model,
        device=cfg.device,
        local_files_only=cfg.local_files_only,
        cache_dir=cfg.cache_dir,
    )

    if args.prompt is not None:
        _run_generation(gen, cfg, args.prompt, outdir)
        return 0

    manifest_path = outdir / "manifest.jsonl"
    assert prompts is not None
    total = len(prompts)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, prompt in enumerate(prompts, start=1):
            run_dir_name = f"{index:04d}"
            run_dir = outdir / run_dir_name
            paths = _run_generation(gen, cfg, prompt, run_dir)

            row = {
                "index": index,
                "prompt": prompt,
                "run_dir": run_dir_name,
                "answer_path": f"{run_dir_name}/{paths['answer_path']}",
                "monologue_path": f"{run_dir_name}/{paths['monologue_path']}",
                "audit_path": f"{run_dir_name}/{paths['audit_path']}",
                "meta_path": f"{run_dir_name}/{paths['meta_path']}",
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{index}/{total}] wrote {Path(args.outdir) / run_dir_name}")

    return 0


def _build_solver_config(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        max_program_depth=args.max_program_depth,
        max_candidates=args.max_candidates,
        beam_width=args.beam_width,
        diversity_penalty=args.diversity_penalty,
        emit_trace=not args.no_trace,
        require_integrity=not args.no_integrity,
        write_candidate_rankings=args.emit_candidate_rankings,
    )


def cmd_solve_task(args: argparse.Namespace) -> int:
    task = load_task(args.task)
    solver = ArcSolver(_build_solver_config(args))
    result = solver.solve_task(task)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_task_artifacts(result, outdir, include_rankings=args.emit_candidate_rankings)
    write_submission([result], outdir / "submission.json")
    return 0


def cmd_solve_dataset(args: argparse.Namespace) -> int:
    tasks = load_tasks_from_dir(args.tasks_dir)
    solver = ArcSolver(_build_solver_config(args))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for task in tasks:
        result = solver.solve_task(task)
        results.append(result)
        write_task_artifacts(
            result,
            outdir / task.task_id,
            include_rankings=args.emit_candidate_rankings,
        )

    write_submission(results, outdir / "submission.json")
    return 0


def cmd_kaggle_submit(args: argparse.Namespace) -> int:
    tasks = load_tasks_from_dir(args.tasks_dir)
    solver = ArcSolver(_build_solver_config(args))
    results = [solver.solve_task(task) for task in tasks]
    write_submission(results, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dualstream",
        description="Dual-Stream Architecture reference implementation",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate answer + monologue evidence for a prompt")
    g.add_argument("--model", default="gpt2", help="HF model name or path")

    prompt_group = g.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="User prompt")
    prompt_group.add_argument(
        "--prompt-file",
        help="UTF-8 text file with one prompt per non-empty line",
    )

    g.add_argument("--outdir", default=".", help="Output directory")
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.add_argument("--top-k", type=int, default=5, help="Top-K evidence tokens per step (pre-sampling)")
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--greedy", action="store_true", help="Disable sampling (argmax)")
    g.add_argument("--seed", type=int, default=None)

    g.add_argument("--include-attn", action="store_true", help="Emit attention summaries (slow)")
    g.add_argument("--include-probes", action="store_true", help="Run concept probes (requires hidden states; slow)")
    g.add_argument("--probe-pack", default=None, help="Path to probe pack JSON")
    g.add_argument("--no-heuristics", action="store_true", help="Disable heuristic concepts")

    g.add_argument("--no-crc32", action="store_true", help="Do not append crc32 to frames")
    g.add_argument("--no-running-hash", action="store_true", help="Disable running hash accumulation")
    g.add_argument("--device", default=None, help="Override device, e.g. cpu/cuda")
    g.add_argument("--offline", action="store_true", help="Use local HF cache only (no network)")
    g.add_argument("--cache-dir", default=None, help="Override HF cache directory")
    g.set_defaults(func=cmd_generate)

    def add_solver_flags(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--max-program-depth", type=int, default=2)
        parser.add_argument("--max-candidates", type=int, default=128)
        parser.add_argument("--beam-width", type=int, default=24)
        parser.add_argument("--diversity-penalty", type=float, default=0.10)
        parser.add_argument("--no-trace", action="store_true")
        parser.add_argument("--no-integrity", action="store_true")
        parser.add_argument("--emit-candidate-rankings", action="store_true")

    st = sub.add_parser("solve-task", help="Solve a single ARC task JSON and emit sidecar artifacts")
    st.add_argument("--task", required=True, help="Path to ARC task JSON file")
    st.add_argument("--outdir", required=True, help="Output directory")
    add_solver_flags(st)
    st.set_defaults(func=cmd_solve_task)

    sd = sub.add_parser("solve-dataset", help="Solve all ARC tasks in a directory and emit submission + artifacts")
    sd.add_argument("--tasks-dir", required=True, help="Directory containing ARC task JSON files")
    sd.add_argument("--outdir", required=True, help="Output directory")
    add_solver_flags(sd)
    sd.set_defaults(func=cmd_solve_dataset)

    ks = sub.add_parser("kaggle-submit", help="Build Kaggle-compatible submission.json from a task directory")
    ks.add_argument("--tasks-dir", required=True, help="Directory containing ARC task JSON files")
    ks.add_argument("--output", required=True, help="Path to submission.json")
    add_solver_flags(ks)
    ks.set_defaults(func=cmd_kaggle_submit)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
