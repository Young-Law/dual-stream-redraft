from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .audit import coherence_outcome, coherence_audit
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
    compact_path = outdir / "compact_evidence.dsae"

    answer_path.write_text(result["answer"], encoding="utf-8")
    monologue_path.write_text(monologue_text, encoding="utf-8")

    with monologue_jsonl_path.open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr.to_dict(), ensure_ascii=False) + "\n")

    with monologue_bin_path.open("wb") as f:
        for fb in result["frame_bytes"]:
            f.write(fb)

    compact_bytes = result.get("compact_evidence_bytes")
    frame_token_count = len(frames)
    answer_token_count = len(result.get("answer_token_ids", [])) if result.get("answer_token_ids") is not None else frame_token_count
    meta = {
        "prompt_nonce": result["prompt_nonce"],
        "model": result["model"],
        "running_hash": result["running_hash"],
        "answer_token_count": answer_token_count,
        "frame_token_count": frame_token_count,
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
            "audit_mode": cfg.audit_mode,
            "poc_mode": cfg.poc_mode,
            "randomized_audit": cfg.randomized_audit,
            "audit_nonce": cfg.audit_nonce,
            "entropy_threshold": cfg.entropy_threshold,
            "refusal_mass_threshold": cfg.refusal_mass_threshold,
            "risk_threshold_review": cfg.risk_threshold_review,
            "risk_threshold_fail": cfg.risk_threshold_fail,
            "max_red_retries": cfg.max_red_retries,
            "fallback_strategy": cfg.fallback_strategy,
            "selective_retention": cfg.selective_retention,
            "evidence_profile": getattr(cfg, "evidence_profile", "DSA-CI-Lite"),
            "compact_evidence": getattr(cfg, "compact_evidence", False),
            "adaptive_k": getattr(cfg, "adaptive_k", False),
            "max_adaptive_k": getattr(cfg, "max_adaptive_k", None),
            "chunk_token_capacity": getattr(cfg, "chunk_token_capacity", 256),
            "compact_wire_version": getattr(cfg, "compact_wire_version", 0x0303),
        },
    }
    if compact_bytes is not None:
        compact_path.write_bytes(compact_bytes)
        meta["compact_evidence_path"] = compact_path.name
        meta["compact_evidence_sha256"] = hashlib.sha256(compact_bytes).hexdigest()
        meta["compact_evidence_token_count"] = frame_token_count
    if result.get("fallback_text"):
        fallback_path = outdir / "fallback.txt"
        fallback_path.write_text(result["fallback_text"], encoding="utf-8")
        meta["fallback_path"] = fallback_path.name
        meta["fallback_evidence_bound"] = False
        meta["original_output_blocked"] = True
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    try:
        audit = coherence_outcome(result["answer"], frames, decode_token=lambda tid: gen.tokenizer.decode([tid]), risk_threshold_review=cfg.risk_threshold_review, risk_threshold_fail=cfg.risk_threshold_fail)
        audit_payload = audit.to_dict()
    except Exception:
        findings = coherence_audit(result["answer"], frames, decode_token=lambda tid: gen.tokenizer.decode([tid]))
        audit_payload = {"outcome": "REVIEW", "findings": [f.__dict__ for f in findings], "max_severity": 0.5, "audit_tier": "tier1", "fallback_recommended": False}
    audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    paths = {
        "answer_path": answer_path.name,
        "monologue_path": monologue_path.name,
        "audit_path": audit_path.name,
        "meta_path": meta_path.name,
    }
    if compact_bytes is not None:
        paths["compact_evidence_path"] = compact_path.name
    return paths


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
        audit_mode=getattr(args, "audit_mode", "tiered"),
        poc_mode=getattr(args, "poc_mode", "none"),
        randomized_audit=getattr(args, "randomized_audit", False),
        audit_nonce=getattr(args, "audit_nonce", None),
        entropy_threshold=getattr(args, "entropy_threshold", 4.0),
        refusal_mass_threshold=getattr(args, "refusal_mass_threshold", 0.20),
        risk_threshold_review=getattr(args, "risk_threshold_review", 0.45),
        risk_threshold_fail=getattr(args, "risk_threshold_fail", 0.70),
        max_red_retries=getattr(args, "max_red_retries", 1),
        fallback_strategy=getattr(args, "fallback_strategy", "canned_refusal"),
        selective_retention=not getattr(args, "no_selective_retention", False),
        evidence_profile=getattr(args, "evidence_profile", "DSA-CI-Lite"),
        compact_evidence=getattr(args, "compact_evidence", False),
        adaptive_k=getattr(args, "adaptive_k", False),
        max_adaptive_k=getattr(args, "max_adaptive_k", None),
        chunk_token_capacity=getattr(args, "chunk_token_capacity", 256),
        compact_wire_version=int(getattr(args, "compact_wire_version", "0x0303"), 0),
    )

    if getattr(cfg, "compact_evidence", False):
        from .evidence_profile import get_evidence_profile
        profile = get_evidence_profile(cfg.evidence_profile)
        if cfg.top_k < profile.base_k:
            cfg.top_k = profile.base_k

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
            if "compact_evidence_path" in paths:
                row["compact_evidence_path"] = f"{run_dir_name}/{paths['compact_evidence_path']}"
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


def cmd_retention_issue(args: argparse.Namespace) -> int:
    import hashlib
    from .retention_lifecycle import issue_retention_requirement

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"ERROR: artifact not found: {artifact_path}")
        return 1

    artifact_bytes = artifact_path.read_bytes()
    artifact_id = hashlib.sha256(artifact_bytes).hexdigest()

    issuer_key = args.issuer_key.encode("utf-8")
    req = issue_retention_requirement(
        artifact_id=artifact_id,
        profile_name=args.profile,
        issuer_id=args.issuer_id,
        issuer_key=issuer_key,
        min_retention_days=args.min_retention_days,
        max_artifact_bytes=args.max_artifact_bytes,
    )

    output = {
        "artifact_id": artifact_id,
        "profile_name": req.profile_name,
        "issuer_id": req.issuer_id,
        "min_retention_days": req.min_retention_days,
        "max_artifact_bytes": req.max_artifact_bytes,
        "issued_at": req.issued_at,
        "expires_at": req.expires_at,
        "nonce": req.nonce,
        "requirement_hash": req.hash().hex(),
        "signature": req.signature.hex(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def cmd_retention_verify_chain(args: argparse.Namespace) -> int:
    import hashlib
    from .retention_lifecycle import (
        RetentionRequirement,
        RetentionReceipt,
        verify_receipt_chain,
    )

    # Load requirement JSON
    req_path = Path(args.requirement)
    if not req_path.exists():
        print(f"ERROR: requirement file not found: {req_path}")
        return 1
    req_data = json.loads(req_path.read_text(encoding="utf-8"))
    requirement = RetentionRequirement(
        artifact_id=req_data["artifact_id"],
        profile_name=req_data["profile_name"],
        issuer_id=req_data["issuer_id"],
        min_retention_days=req_data["min_retention_days"],
        max_artifact_bytes=req_data["max_artifact_bytes"],
        issued_at=req_data.get("issued_at"),
        expires_at=req_data.get("expires_at"),
        nonce=req_data.get("nonce", ""),
        signature=bytes.fromhex(req_data.get("signature", "")),
    )

    # Load receipt JSON
    rcpt_path = Path(args.receipt)
    if not rcpt_path.exists():
        print(f"ERROR: receipt file not found: {rcpt_path}")
        return 1
    rcpt_data = json.loads(rcpt_path.read_text(encoding="utf-8"))
    receipt = RetentionReceipt(
        requirement_hash=bytes.fromhex(rcpt_data["requirement_hash"]),
        artifact_content_hash=bytes.fromhex(rcpt_data["artifact_content_hash"]),
        storage_backend=rcpt_data["storage_backend"],
        stored_bytes=rcpt_data["stored_bytes"],
        stored_at=rcpt_data.get("stored_at"),
        validator_id=rcpt_data.get("validator_id", ""),
        nonce=rcpt_data.get("nonce", ""),
        signature=bytes.fromhex(rcpt_data.get("signature", "")),
    )

    # Load artifact
    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"ERROR: artifact not found: {artifact_path}")
        return 1
    artifact_bytes = artifact_path.read_bytes()

    issuer_key = args.issuer_key.encode("utf-8")
    validator_key = args.validator_key.encode("utf-8")

    result = verify_receipt_chain(requirement, receipt, artifact_bytes, issuer_key, validator_key)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["valid"]:
        print(f"CHAIN VALID: requirement → receipt → artifact verified")
    else:
        print(f"CHAIN INVALID: {'; '.join(result['errors'])}")

    return 0 if result["valid"] else 2


def cmd_retention_challenge(args: argparse.Namespace) -> int:
    """Issue a possession challenge, auto-respond, and verify the result."""
    import hashlib
    from .challenge import (
        issue_possession_challenge,
        respond_to_challenge,
        verify_possession_challenge,
    )

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"ERROR: artifact not found: {artifact_path}")
        return 1

    artifact_bytes = artifact_path.read_bytes()
    artifact_id = hashlib.sha256(artifact_bytes).hexdigest()

    challenger_key = args.challenger_key.encode("utf-8")
    responder_key = args.responder_key.encode("utf-8")

    # Issue challenge
    challenge = issue_possession_challenge(
        artifact_id=artifact_id,
        artifact_bytes=artifact_bytes,
        challenger_key=challenger_key,
        start_offset=args.start_offset,
        end_offset=args.end_offset,
        ttl_seconds=args.ttl,
    )

    # Auto-respond (simulates holder proving possession)
    response = respond_to_challenge(
        challenge=challenge,
        artifact_bytes=artifact_bytes,
        responder_id=args.responder_id,
        responder_key=responder_key,
    )

    # Verify
    result = verify_possession_challenge(
        challenge=challenge,
        response=response,
        challenger_key=challenger_key,
    )

    if args.json:
        output = {
            "challenge_id": challenge.challenge_id,
            "artifact_id": challenge.artifact_id,
            "start_offset": challenge.start_offset,
            "end_offset": challenge.end_offset,
            "valid": result["valid"],
            "errors": result["errors"],
            "signature": challenge.signature.hex(),
            "response_signature": response.signature.hex(),
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    elif result["valid"]:
        print(
            f"CHALLENGE VERIFIED: artifact {artifact_id[:16]}… "
            f"range [{challenge.start_offset}:{challenge.end_offset}] "
            f"possession confirmed"
        )
    else:
        print(f"CHALLENGE FAILED: {'; '.join(result['errors'])}")

    return 0 if result["valid"] else 2



def cmd_verify_evidence_budget(args: argparse.Namespace) -> int:
    from .verifier import verify_evidence_artifact
    vkey = args.verifier_key.encode("utf-8") if getattr(args, "verifier_key", None) is not None else None
    report = verify_evidence_artifact(args.artifact, profile=args.profile, ci_mode=args.ci_mode, strict_profile_budget=args.strict_profile_budget, enforce_rss_budget=args.enforce_rss_budget, verifier_key=vkey)
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report.ok:
        suffix = "" if report.budget_status == "pass" else f" (budget_status={report.budget_status})"
        print(f"PASS {report.profile_id}: {report.token_count} tokens, {report.raw_bytes_per_token:.3f} bytes/token{suffix}")
    else:
        status = f"budget_status={report.budget_status}"
        detail = "; ".join(report.errors) if report.errors else report.verification_outcome
        print(f"FAIL {status}: {detail}")
    return 0 if report.ok else 2


def cmd_conformance(args: argparse.Namespace) -> int:
    from .compact_evidence import VERSION_V33, decode_compact_sequence, encode_compact_sequence, verify_keyed_replay

    key = b"codex-v210-conformance-key"
    tokens = []
    for i in range(128):
        ids = list(range(i * 16, i * 16 + 10))
        chosen = ids[6] if i % 17 == 0 else ids[0]
        tokens.append({
            "chosen_id": chosen,
            "topk_ids": ids,
            "topk_scores": [1.0 - (j / 20) for j in range(10)],
            "trigger_flags": 4 if i == 11 else 0,
        })
    artifact = encode_compact_sequence(
        tokens,
        profile="DSA-CI-Lite",
        sequence_id=210,
        adaptive_k=True,
        wire_version=VERSION_V33,
        audit_key=key,
        audit_key_id=7,
        stochastic_rate_ppm=5000,
    )
    decoded = decode_compact_sequence(artifact)
    verify_keyed_replay(decoded, {7: key})
    summary = {
        "wire_version": "0x0303",
        "outcome": "LOCAL_PASS",
        "tokens": decoded["header"].token_count,
        "chunks": decoded["manifest"].chunk_count,
        "raw_bytes_per_token": len(artifact) / decoded["header"].token_count,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "DSA v2.10 Phase 1 conformance: PASS "
            f"({summary['tokens']} tokens, {summary['chunks']} chunks, "
            f"wire {summary['wire_version']})"
        )
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
    g.add_argument("--audit-mode", choices=["off","cheap","tiered","full"], default="tiered")
    g.add_argument("--poc-mode", choices=["none","level0_plumbing","level1_sycophancy_proxy","level2_monitor_evasion_proxy"], default="none")
    g.add_argument("--randomized-audit", action="store_true")
    g.add_argument("--audit-nonce", type=int, default=None)
    g.add_argument("--entropy-threshold", type=float, default=4.0)
    g.add_argument("--refusal-mass-threshold", type=float, default=0.20)
    g.add_argument("--risk-threshold-review", type=float, default=0.45)
    g.add_argument("--risk-threshold-fail", type=float, default=0.70)
    g.add_argument("--max-red-retries", type=int, default=1)
    g.add_argument("--fallback-strategy", choices=["canned_refusal","safe_override","abort"], default="canned_refusal")
    g.add_argument("--no-selective-retention", action="store_true")
    g.add_argument("--evidence-profile", default="DSA-CI-Lite", choices=["DSA-CI-Lite", "DSA-CI-Standard", "DSA-Deep", "DSA-Forensic"])
    g.add_argument("--compact-evidence", action="store_true")
    g.add_argument("--adaptive-k", action="store_true")
    g.add_argument("--max-adaptive-k", type=int, default=None)
    g.add_argument("--chunk-token-capacity", type=int, default=256)
    g.add_argument("--compact-wire-version", default="0x0303", choices=["0x0302", "0x0303"])

    g.set_defaults(func=cmd_generate)

    vb = sub.add_parser("verify-evidence-budget", help="Verify compact evidence profile budget and retention floor")
    vb.add_argument("--artifact", required=True)
    vb.add_argument("--profile", required=True, choices=["DSA-CI-Lite", "DSA-CI-Standard", "DSA-Deep", "DSA-Forensic"])
    vb.add_argument("--ci-mode", required=True)
    vb.add_argument("--json", action="store_true")
    vb.add_argument("--strict-profile-budget", action="store_true")
    vb.add_argument("--enforce-rss-budget", action="store_true")
    vb.add_argument("--verifier-key", default=None, help="Symmetric UTF-8 key to sign the Verifier Work Certificate")
    vb.set_defaults(func=cmd_verify_evidence_budget)

    conf = sub.add_parser("conformance", help="Run deterministic DSA conformance checks")
    conf.add_argument("version", choices=["v2.10"])
    conf.add_argument("--json", action="store_true")
    conf.set_defaults(func=cmd_conformance)

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

    # P8: Retention lifecycle subcommands
    ri = sub.add_parser("retention-issue", help="Issue a retention requirement for an artifact")
    ri.add_argument("--artifact", required=True, help="Path to the artifact file")
    ri.add_argument("--profile", default="DSA-CI-Lite", choices=["DSA-CI-Lite", "DSA-CI-Standard", "DSA-Deep", "DSA-Forensic"], help="Evidence profile name")
    ri.add_argument("--issuer-id", default="local-governance", help="Issuer identifier")
    ri.add_argument("--issuer-key", required=True, help="Symmetric UTF-8 issuer key for HMAC signing")
    ri.add_argument("--min-retention-days", type=int, default=90, help="Minimum retention period in days")
    ri.add_argument("--max-artifact-bytes", type=int, default=10_000_000, help="Maximum artifact size in bytes")
    ri.set_defaults(func=cmd_retention_issue)

    rvc = sub.add_parser("retention-verify-chain", help="Verify a full requirement→receipt→artifact chain")
    rvc.add_argument("--requirement", required=True, help="Path to requirement JSON")
    rvc.add_argument("--receipt", required=True, help="Path to receipt JSON")
    rvc.add_argument("--artifact", required=True, help="Path to the artifact file")
    rvc.add_argument("--issuer-key", required=True, help="Symmetric UTF-8 issuer key")
    rvc.add_argument("--validator-key", required=True, help="Symmetric UTF-8 validator key")
    rvc.add_argument("--json", action="store_true", help="Output structured JSON")
    rvc.set_defaults(func=cmd_retention_verify_chain)

    rc = sub.add_parser("retention-challenge", help="Issue and verify a possession challenge for an artifact")
    rc.add_argument("--artifact", required=True, help="Path to the artifact file")
    rc.add_argument("--challenger-key", required=True, help="Symmetric UTF-8 key for the challenger")
    rc.add_argument("--responder-key", required=True, help="Symmetric UTF-8 key for the responder")
    rc.add_argument("--responder-id", default="local-holder", help="Responder identifier")
    rc.add_argument("--start-offset", type=int, default=0, help="Byte range start")
    rc.add_argument("--end-offset", type=int, default=None, help="Byte range end (default: full artifact)")
    rc.add_argument("--ttl", type=int, default=300, help="Challenge time-to-live in seconds")
    rc.add_argument("--json", action="store_true", help="Output structured JSON")
    rc.set_defaults(func=cmd_retention_challenge)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
