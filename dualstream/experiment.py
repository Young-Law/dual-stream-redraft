from __future__ import annotations

import argparse, gzip, hashlib, json, os, platform, statistics, subprocess, sys, time, uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .compact_evidence import encode_compact_sequence
from .evidence_profile import PROFILES, get_evidence_profile
from .verifier import verify_evidence_artifact

PROTOCOL_REQUIRED = ["protocol_id","dsa_specification_version","schema_version","schema_hash","model_identifiers","tokenizer_identifiers","evidence_profile_identifiers","verifier_version","verifier_budget_profile","retention_floor_policy","probe_pack_identifiers","benchmark_identifiers","calibration_split","test_split","random_seeds","decoding_parameters","hardware","primary_endpoint","secondary_endpoints","baseline_conditions","ablation_conditions","exclusion_criteria","failure_handling_rules","statistical_analysis_method","confidence_interval_method","protocol_freeze_timestamp","implementation_validity_hypotheses","signal_validity_hypotheses"]

@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str; purpose: str; hypothesis_ids: list[str]; required_inputs: list[str]; supported_models: list[str]; configuration_schema: dict[str, Any]; output_schema: dict[str, Any]; metrics: list[str]; expected_failure_modes: list[str]; reproducibility_requirements: list[str]

REGISTRY = {
 "exp01_conformance": ExperimentDefinition("exp01_conformance","DSA systems conformance and overhead evaluation using synthetic controlled replay.",["evidence_completeness","reconstruction_correctness","corruption_rejection","storage_overhead","verifier_latency","verifier_memory","retention_floor_enforcement","adaptive_evidence_behavior"],[],["synthetic-token-stream"],{"token_count":"integer >= 1","profiles":"list of supported profile identifiers"},{"metrics.json":"aggregate metrics","events.jsonl":"per-condition events","failures.jsonl":"negative fixture outcomes"},["generated_token_count","evidence_record_count","evidence_completeness_rate","raw_bytes_per_token","compressed_bytes_per_token","encoding_latency","generation_throughput","generation_throughput_delta_vs_disabled","verifier_reconstruction_p50","verifier_reconstruction_p95","verifier_reconstruction_max","verifier_peak_rss","reconstruction_throughput","reconstruction_error_rate","adaptive_record_fraction","maximum_effective_k","retention_floor_margin","malformed_artifact_rejection_rate","malformed_artifact_false_acceptance_rate"],["schema_mismatch","corruption_accepted","budget_exceeded"],["fixed seed","recorded git commit","artifact hashes"]),
 "exp02_signal_validity": ExperimentDefinition("exp02_signal_validity","Controlled signal-validity interface; requires external labeled benchmark and model telemetry.",["failure_detection","false_positive_burden","silent_deception_rate"],["benchmark_path","label_field","model_id","tokenizer_id"],["external"],{}, {}, ["AUROC","AUPRC","precision","recall","F1","false_positive_rate","silent_deception_rate"],["missing_dataset","missing_activation_hook","missing_label_field"],["calibration/test split separation"]),
 "exp03_ablations": ExperimentDefinition("exp03_ablations","Ablation result schema and validation for DSA evidence components.",["marginal_value"],["completed_signal_validity_runs"],["external"],{}, {}, ["marginal_gain","storage_cost","compute_cost"],["incompatible_runs"],["source-run mapping"]),
 "exp04_monitor_evasion": ExperimentDefinition("exp04_monitor_evasion","Safe monitor-aware adversarial evaluation interface for externally supplied attacker checkpoints.",["monitor_evasion_cost"],["attacker_checkpoint","sandbox_benchmark"],["external"],{}, {}, ["hidden_behavior_success","silent_deception_rate","optimization_steps","attack_compute","monitor_score","benign_utility"],["missing_attacker_checkpoint"],["sandboxed evaluation only"]),
 "exp05_enforcement": ExperimentDefinition("exp05_enforcement","Verifier enforcement and fail-closed result schema.",["fail_closed_behavior"],["evidence_artifacts"],["synthetic-token-stream","external"],{}, {}, ["reject_rate","false_acceptance_rate"],["unsupported_assurance_claim"],["hash-verified artifacts"]),
}

def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def validate_protocol(path: str | Path) -> dict[str, Any]:
    p=Path(path); data=_json_load(p)
    missing=[k for k in PROTOCOL_REQUIRED if k not in data]
    if missing: raise ValueError(f"protocol missing required fields: {', '.join(missing)}")
    if data.get("dsa_specification_version") == "v2.10" and "blocked" in data.get("schema_version",""):
        raise ValueError("protocol cannot claim full v2.10 with blocked schema version")
    return data

def list_experiments() -> list[dict[str, Any]]: return [asdict(v) for v in REGISTRY.values()]

def _git_commit() -> str:
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return "unknown"

def _git_status() -> str:
    try: return subprocess.check_output(["git","status","--porcelain"], text=True).strip()
    except Exception: return "unknown"

def _dependency_versions() -> dict[str,str]: return {"python": sys.version.split()[0]}

def _make_tokens(n:int, base_k:int, max_k:int, adaptive:bool) -> list[dict[str, Any]]:
    rows=[]
    for i in range(n):
        width=max_k if adaptive and i % 11 == 0 else base_k
        ids=list(range(1000+i*50,1000+i*50+width)); chosen=ids[-1] if width>base_k else ids[0]
        scores=[max(0.0,1.0-(j*0.03)) for j in range(width)]
        rows.append({"token_index":i,"chosen_id":chosen,"topk_ids":ids,"topk_scores":scores})
    return rows

def _corruptions(good: bytes) -> dict[str, bytes]:
    out={"truncated_final_chunk": good[:-1], "invalid_schema_magic": b"BADSCHEM"+good[8:], "invalid_hash_byte_flip": good[:-8]+bytes([good[-8]^1])+good[-7:]}
    return out

def _hash_file(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def run_exp01(protocol_path: Path, out_root: Path, token_count: int=128, repetitions: int=20) -> Path:
    protocol=validate_protocol(protocol_path); started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_basis=json.dumps({"experiment_id":"exp01_conformance","protocol":protocol.get("protocol_id"),"commit":_git_commit(),"seed":protocol.get("random_seeds",[0])[0],"token_count":token_count,"started":started}, sort_keys=True)
    run_id="run-"+uuid.uuid5(uuid.NAMESPACE_URL, run_basis).hex
    run_dir=out_root/run_id; run_dir.mkdir(parents=True, exist_ok=False)
    metrics={"experiment_id":"exp01_conformance","token_count":token_count,"conditions":{}}; events=[]; failures=[]
    disabled_tps=0.0
    for profile in ["DSA-disabled","DSA-CI-Lite","DSA-CI-Lite-adaptive","DSA-CI-Standard","DSA-Deep"]:
        if profile=="DSA-disabled":
            t0=time.perf_counter(); _make_tokens(token_count,1,1,False); elapsed=time.perf_counter()-t0; disabled_tps=token_count/max(elapsed,1e-9)
            metrics["conditions"][profile]={"generated_token_count":token_count,"evidence_record_count":0,"evidence_completeness_rate":0.0,"generation_throughput":disabled_tps,"generation_throughput_delta_vs_disabled":0.0}; continue
        prof_name="DSA-CI-Lite" if profile.endswith("adaptive") else profile; prof=get_evidence_profile(prof_name); adaptive=profile.endswith("adaptive")
        tokens=_make_tokens(token_count, prof.base_k, prof.max_adaptive_k, adaptive)
        t0=time.perf_counter(); blob=encode_compact_sequence(tokens, profile=prof_name, adaptive_k=adaptive, sequence_id=1); enc=time.perf_counter()-t0
        ev_path=run_dir/f"{profile}.dsae"; ev_path.write_bytes(blob)
        samples=[]; report=None
        for _ in range(max(4,repetitions)):
            r=verify_evidence_artifact(ev_path, profile=prof_name, ci_mode="nightly" if prof_name!="DSA-Forensic" else "forensic", enforce_budget=False); samples.append(r.elapsed_seconds); report=r
        assert report is not None
        gz=len(gzip.compress(blob)); p95=statistics.quantiles(samples,n=20,method="inclusive")[18]
        cond={"generated_token_count":token_count,"evidence_record_count":report.token_count,"evidence_completeness_rate":report.token_count/token_count,"raw_bytes_per_token":len(blob)/token_count,"compressed_bytes_per_token":gz/token_count,"encoding_latency":enc,"generation_throughput":token_count/max(enc,1e-9),"generation_throughput_delta_vs_disabled":(token_count/max(enc,1e-9)-disabled_tps)/disabled_tps if disabled_tps else 0.0,"verifier_reconstruction_p50":statistics.median(samples),"verifier_reconstruction_p95":p95,"verifier_reconstruction_max":max(samples),"verifier_peak_rss":report.verifier_peak_rss_bytes,"reconstruction_throughput":report.tokens_reconstructed_per_second,"reconstruction_error_rate":0.0 if report.ok else 1.0,"adaptive_record_fraction":report.adaptive_record_fraction,"maximum_effective_k":report.max_effective_topk,"retention_floor_margin":report.retention_floor_margin_bytes}
        rejects=0; accepts=0
        for name,bad in _corruptions(blob).items():
            bp=run_dir/f"corrupt-{profile}-{name}.dsae"; bp.write_bytes(bad); rr=verify_evidence_artifact(bp, profile=prof_name, ci_mode="nightly", enforce_budget=False); rejected=not rr.ok; rejects+=int(rejected); accepts+=int(not rejected); failures.append({"condition":profile,"corruption":name,"expected":"reject","observed":"reject" if rejected else "accept","errors":rr.errors})
        cond["malformed_artifact_rejection_rate"]=rejects/(rejects+accepts); cond["malformed_artifact_false_acceptance_rate"]=accepts/(rejects+accepts)
        metrics["conditions"][profile]=cond; events.append({"condition":profile,"artifact":ev_path.name,"sha256":_hash_file(ev_path)})
    completed=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest={"run_id":run_id,"experiment_id":"exp01_conformance","protocol_id":protocol["protocol_id"],"started_at":started,"completed_at":completed,"git_commit":_git_commit(),"working_tree_status":_git_status(),"implementation_version":"0.1.0-v2.10-partial","schema_id":protocol["schema_version"],"schema_hash":protocol["schema_hash"],"model_id":"synthetic-token-stream","model_hash_or_revision":"not_applicable","tokenizer_id":protocol["tokenizer_identifiers"][0],"tokenizer_hash_or_revision":"synthetic-v1","probe_pack_id":"none","probe_pack_hash":hashlib.sha256(b"none").hexdigest(),"benchmark_id":"synthetic-controlled-replay","benchmark_version":"1","split":"synthetic_conformance","seed":protocol["random_seeds"][0],"hardware":platform.machine(),"operating_system":platform.platform(),"runtime_version":sys.version,"dependency_versions":_dependency_versions(),"decoding_configuration":protocol["decoding_parameters"],"evidence_profile":"multiple","verifier_budget":protocol["verifier_budget_profile"],"retention_policy":protocol["retention_floor_policy"],"command":" ".join(sys.argv),"exit_status":0}
    (run_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8"); (run_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True),encoding="utf-8")
    (run_dir/"events.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in events),encoding="utf-8"); (run_dir/"failures.jsonl").write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in failures),encoding="utf-8")
    (run_dir/"environment.json").write_text(json.dumps({"platform":platform.platform(),"python":sys.version,"cwd":os.getcwd()},indent=2),encoding="utf-8")
    hashes={p.name:_hash_file(p) for p in run_dir.iterdir() if p.is_file() and p.name!="artifact_hashes.json"}; (run_dir/"artifact_hashes.json").write_text(json.dumps(hashes,indent=2,sort_keys=True),encoding="utf-8")
    return run_dir

def verify_hashes(run_dir: str | Path) -> bool:
    p=Path(run_dir); hashes=_json_load(p/"artifact_hashes.json")
    bad=[name for name,h in hashes.items() if not (p/name).exists() or _hash_file(p/name)!=h]
    if bad: raise ValueError("artifact hash mismatch: "+", ".join(bad))
    return True

def regenerate_summaries(raw_root: str|Path="results/raw", out: str|Path="results/summaries/exp01_summary.json") -> Path:
    rows=[]
    for m in Path(raw_root).glob("run-*/metrics.json"):
        verify_hashes(m.parent); rows.append({"run_id":m.parent.name,"metrics":_json_load(m)})
    outp=Path(out); outp.parent.mkdir(parents=True, exist_ok=True); outp.write_text(json.dumps({"source_runs":[r["run_id"] for r in rows],"runs":rows},indent=2,sort_keys=True),encoding="utf-8"); return outp

def main_cli(args: argparse.Namespace) -> int:
    if args.experiment_cmd=="list": print(json.dumps(list_experiments(),indent=2)); return 0
    if args.experiment_cmd=="validate-protocol": validate_protocol(args.protocol); print("PASS protocol valid"); return 0
    if args.experiment_cmd=="run":
        if args.experiment not in REGISTRY: raise SystemExit(f"unknown experiment ID: {args.experiment}")
        if args.experiment != "exp01_conformance": raise SystemExit(f"experiment {args.experiment} requires external inputs and is not executable without them")
        rd=run_exp01(Path(args.protocol), Path(args.outdir), args.token_count, args.repetitions); print(rd); return 0
    if args.experiment_cmd=="verify-hashes": verify_hashes(args.run_dir); print("PASS artifact hashes valid"); return 0
    if args.experiment_cmd=="summarize": print(regenerate_summaries(args.raw_root,args.output)); return 0
    if args.experiment_cmd=="tables":
        summary=_json_load(Path(args.summary)); out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({"source_runs":summary.get("source_runs",[]),"tables":summary.get("runs",[])},indent=2),encoding="utf-8"); print(out); return 0
    raise SystemExit("unknown experiment command")
