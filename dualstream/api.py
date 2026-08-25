"""DualStream Browser API — FastAPI application.


Exposes v2.10 subsystems (envelopes, triggers, retention, evidence,
verifier, tension map) alongside the original generation/ARC/script job APIs.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .service import DualStreamService

app = FastAPI(title="DualStream Browser API", version="2.10.0")
service = DualStreamService()
WEB_ROOT = Path(__file__).resolve().parent / "web"

# In-memory envelope session store for demo purposes
_envelope_sessions: dict[str, Any] = {}

if WEB_ROOT.exists():
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")


# ======================================================================
# Static files & health
# ======================================================================

@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "2.10.0"}


@app.get("/ui/status")
def ui_status() -> dict[str, Any]:
    return {
        "offline_default": True,
        "version": "2.10.0",
        "envelope_sessions": len(_envelope_sessions),
    }


# ======================================================================
# Original job APIs (generation, ARC, scripts)
# ======================================================================

@app.post("/preflight/{kind}")
def preflight(kind: str, payload: dict) -> dict:
    normalized = kind.replace("-", "_")
    mapping = {
        "generate": "generate",
        "arc_solve_task": "arc_solve_task",
        "arc_solve_dataset": "arc_solve_dataset",
        "kaggle_submit": "kaggle_submit",
    }
    resolved = mapping.get(normalized)
    if not resolved:
        raise HTTPException(status_code=400, detail=f"Unsupported preflight kind: {kind}")
    result = service.preflight(resolved, payload)
    return result


@app.post("/generate")
def generate(payload: dict) -> dict:
    job = service.start_generate(payload)
    return {"job_id": job.id, "status": job.status}


@app.post("/arc/solve-task")
def arc_solve_task(payload: dict) -> dict:
    job = service.start_arc_solve_task(payload)
    return {"job_id": job.id, "status": job.status}


@app.post("/arc/solve-dataset")
def arc_solve_dataset(payload: dict) -> dict:
    job = service.start_arc_solve_dataset(payload)
    return {"job_id": job.id, "status": job.status}


@app.post("/arc/kaggle-submit")
def arc_kaggle_submit(payload: dict) -> dict:
    job = service.start_kaggle_submit(payload)
    return {"job_id": job.id, "status": job.status}


@app.get("/scripts")
def list_scripts() -> list[dict]:
    return service.list_scripts()


@app.post("/scripts/run")
def run_script(payload: dict) -> dict:
    job = service.start_script(payload)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs")
def list_jobs() -> list[dict]:
    return [asdict(job) for job in service.list_jobs()]


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(job)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, bool]:
    ok = service.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


@app.get("/artifacts/{job_id}")
def get_artifacts(job_id: str) -> list[dict]:
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.output_dir:
        return []
    from .artifacts import discover_artifacts

    artifacts = discover_artifacts(job.output_dir)
    for artifact in artifacts:
        rel_path = quote(str(artifact["relative_path"]))
        artifact["url"] = f"/artifacts/{job_id}/file/{rel_path}"
    return artifacts


@app.get("/artifacts/{job_id}/file/{artifact_path:path}")
def get_artifact_file(job_id: str, artifact_path: str) -> FileResponse:
    job = service.get_job(job_id)
    if not job or not job.output_dir:
        raise HTTPException(status_code=404, detail="Job or output directory not found")
    root_dir = Path(job.output_dir).resolve()
    target = (root_dir / artifact_path).resolve()
    if root_dir not in target.parents and target != root_dir:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=target)


# ======================================================================
# v2.10 — Streaming Envelope Session API (§5.3)
# ======================================================================

@app.post("/v210/envelope/create")
def envelope_create(payload: dict) -> dict:
    """Open a new streaming-space envelope session."""
    from .envelope import EnvelopeSession, EnvelopePolicy

    session_id = payload.get("session_id", f"web-{int(time.time()*1000)}")
    issuer_key = (payload.get("issuer_key") or "").encode() or None
    require_sig = payload.get("require_signature", True)
    if require_sig and issuer_key is None:
        issuer_key = b"web-ui-default-key"

    policy = EnvelopePolicy(
        max_token_count=int(payload.get("max_token_count", 100_000)),
        max_interim_commits=int(payload.get("max_interim_commits", 100)),
        commit_interval=int(payload.get("commit_interval", 100)),
        ttl_seconds=float(payload.get("ttl_seconds", 3600.0)),
        require_signature=require_sig,
        enforce_canonical_order=bool(payload.get("enforce_canonical_order", True)),
    )
    try:
        session = EnvelopeSession.open(
            session_id=session_id,
            profile_id=payload.get("profile_id", "DSA-CI-Lite"),
            issuer_id=payload.get("issuer_id", "web-ui"),
            retention_policy_id=payload.get("retention_policy_id", "local-floor-v1"),
            issuer_key=issuer_key,
            assurance_class=payload.get("assurance_class", "DSA-R"),
            benchmark_id=payload.get("benchmark_id", ""),
            prompt_hash=payload.get("prompt_hash", ""),
            base_k=int(payload.get("base_k", 3)),
            max_adaptive_k=int(payload.get("max_adaptive_k", 10)),
            policy=policy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _envelope_sessions[session_id] = {
        "session": session,
        "token_records": [],
        "created_at": time.time(),
    }
    return {
        "session_id": session_id,
        "state": session.state.value,
        "header": {
            "session_id": session.header.session_id,
            "profile_id": session.header.profile_id,
            "issuer_id": session.header.issuer_id,
            "assurance_class": session.header.assurance_class,
            "opened_at": session.header.opened_at,
            "base_k": session.header.base_k,
            "max_adaptive_k": session.header.max_adaptive_k,
        },
        "policy": {
            "max_token_count": policy.max_token_count,
            "commit_interval": policy.commit_interval,
            "ttl_seconds": policy.ttl_seconds,
        },
    }


@app.post("/v210/envelope/ingest")
def envelope_ingest(payload: dict) -> dict:
    """Ingest token records into an open envelope session."""
    session_id = payload.get("session_id", "")
    store = _envelope_sessions.get(session_id)
    if not store or store["session"].state.value != "open":
        raise HTTPException(status_code=400, detail="Session not found or not open")

    session: Any = store["session"]
    tokens = payload.get("tokens", [])
    ingested = 0
    last_commitment = None
    errors: list[str] = []

    for tok in tokens:
        try:
            session.ingest(tok)
            store["token_records"].append(tok)
            ingested += 1
        except ValueError as exc:
            errors.append(f"Token {tok.get('token_index', '?')}: {exc}")
            break

    # Auto-create an interim commitment after batch
    if ingested > 0 and session.state.value == "open":
        try:
            last_commitment = {
                "token_index": session.token_count - 1,
                "token_count": session.token_count,
                "cumulative_hash": session.interim_commitments[-1].cumulative_hash if session.interim_commitments else "",
                "commitment_count": len(session.interim_commitments),
            }
        except (IndexError, AttributeError):
            pass

    return {
        "session_id": session_id,
        "state": session.state.value,
        "ingested": ingested,
        "total_tokens": session.token_count,
        "last_commitment": last_commitment,
        "errors": errors,
    }


@app.post("/v210/envelope/seal")
def envelope_seal(payload: dict) -> dict:
    """Seal an open envelope session."""
    from .envelope import verify_envelope

    session_id = payload.get("session_id", "")
    store = _envelope_sessions.get(session_id)
    if not store or store["session"].state.value != "open":
        raise HTTPException(status_code=400, detail="Session not found or not open")

    session: Any = store["session"]
    artifact_text = payload.get("artifact_text", "")
    artifact_bytes = artifact_text.encode("utf-8") if artifact_text else None

    try:
        sealed = session.seal(final_artifact_bytes=artifact_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Immediately verify
    issuer_key = payload.get("issuer_key") or ""
    verify_result = verify_envelope(
        sealed,
        issuer_key=issuer_key.encode() if issuer_key else None,
        artifact_bytes=artifact_bytes,
    )

    return {
        "session_id": session_id,
        "state": "sealed",
        "total_tokens": sealed.total_tokens,
        "final_cumulative_hash": sealed.final_cumulative_hash,
        "artifact_content_hash": sealed.artifact_content_hash,
        "seal_nonce": sealed.seal_nonce,
        "sealed_at": sealed.sealed_at,
        "interim_commitments": [
            {
                "token_index": ic.token_index,
                "token_count": ic.token_count,
                "cumulative_hash": ic.cumulative_hash[:24] + "…",
                "timestamp": ic.timestamp,
            }
            for ic in sealed.interim_commitments
        ],
        "verification": verify_result,
    }


@app.get("/v210/envelope/sessions")
def envelope_list_sessions() -> list[dict]:
    """List active/completed envelope sessions."""
    result = []
    for sid, store in _envelope_sessions.items():
        session = store["session"]
        result.append({
            "session_id": sid,
            "state": session.state.value,
            "token_count": session.token_count,
            "created_at": store["created_at"],
            "profile_id": session.header.profile_id,
        })
    return result


# ======================================================================
# v2.10 — Trigger Pipeline API (§6.2)
# ======================================================================

@app.post("/v210/trigger/evaluate")
def trigger_evaluate(payload: dict) -> dict:
    """Evaluate a trigger pipeline configuration against a context."""
    from .triggers import (
        ThresholdTrigger, ThresholdOperator,
        SequenceTrigger, SequenceMode,
        CompositeTrigger, CompositeOperator,
        TriggerPipeline,
    )
    from .compact_evidence import TRIGGER_RANK, TRIGGER_STOCHASTIC, TRIGGER_HISTORY, TRIGGER_CANARY, TRIGGER_ESCALATION

    TRIGGER_FLAGS = {
        "rank": TRIGGER_RANK, "stochastic": TRIGGER_STOCHASTIC,
        "history": TRIGGER_HISTORY, "canary": TRIGGER_CANARY, "escalation": TRIGGER_ESCALATION,
    }

    trigger_defs = payload.get("triggers", [])
    context = payload.get("context", {})
    triggers = []

    for td in trigger_defs:
        ttype = td.get("type", "threshold")
        try:
            if ttype == "threshold":
                flag_name = td.get("flag", "history")
                triggers.append(ThresholdTrigger(
                    name=td.get("name", ""),
                    signal=td.get("signal", "entropy"),
                    operator=ThresholdOperator(td.get("operator", ">=")),
                    threshold=float(td.get("threshold", 0.0)),
                    trigger_flag=TRIGGER_FLAGS.get(flag_name, TRIGGER_HISTORY),
                ))
            elif ttype == "sequence":
                flag_name = td.get("flag", "history")
                triggers.append(SequenceTrigger(
                    name=td.get("name", ""),
                    mode=SequenceMode(td.get("mode", "consecutive")),
                    signal=td.get("signal", "entropy"),
                    count_threshold=int(td.get("count_threshold", 3)),
                    window_size=int(td.get("window_size", 100)),
                    rate_threshold=float(td.get("rate_threshold", 0.2)),
                    trigger_flag=TRIGGER_FLAGS.get(flag_name, TRIGGER_HISTORY),
                ))
            elif ttype == "composite":
                triggers.append(CompositeTrigger(
                    name=td.get("name", ""),
                    operator=CompositeOperator(td.get("operator", "or")),
                    trigger_flag=TRIGGER_FLAGS.get(td.get("flag", "escalation"), TRIGGER_ESCALATION),
                ))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid trigger '{td.get('name', '?')}': {exc}") from exc

    pipeline = TriggerPipeline(triggers=triggers)

    # Evaluate against each context entry (supports batch)
    contexts = context if isinstance(context, list) else [context]
    results = []
    cumulative_flags = 0
    for ctx in contexts:
        result = pipeline.evaluate(ctx)
        cumulative_flags |= result.trigger_flag
        results.append({
            "token_index": ctx.get("token_index", -1),
            "fired": result.fired,
            "trigger_flag": result.trigger_flag,
            "trigger_flag_hex": f"0x{result.trigger_flag:02x}",
            "reason": result.reason,
            "detail": result.detail,
        })

    return {
        "pipeline_size": len(triggers),
        "contexts_evaluated": len(results),
        "cumulative_flags": cumulative_flags,
        "cumulative_flags_hex": f"0x{cumulative_flags:02x}",
        "results": results,
    }


@app.get("/v210/trigger/signals")
def trigger_signals() -> dict:
    """Return available signal names for trigger configuration."""
    from .triggers import SIGNAL_NAMES, ThresholdOperator, SequenceMode, CompositeOperator
    return {
        "signals": list(SIGNAL_NAMES.keys()),
        "operators": [e.value for e in ThresholdOperator],
        "sequence_modes": [e.value for e in SequenceMode],
        "composite_operators": [e.value for e in CompositeOperator],
        "trigger_flags": {
            "rank": 0x01, "stochastic": 0x02, "history": 0x04,
            "canary": 0x08, "escalation": 0x10,
        },
    }


# ======================================================================
# v2.10 — Retention Pipeline API (§7)
# ======================================================================

@app.post("/v210/retention/pipeline")
def retention_pipeline(payload: dict) -> dict:
    """Run the full retention assurance pipeline."""
    from .retention_manager import RetentionPipeline
    from .storage_validator import LocalFilesystemBackend

    artifact_id = payload.get("artifact_id", f"artifact-{int(time.time()*1000)}")
    content = payload.get("content", "")
    artifact_bytes = content.encode("utf-8") if content else b""
    if not artifact_bytes:
        raise HTTPException(status_code=400, detail="content is required")

    storage_dir = payload.get("storage_dir", "/tmp/dsa-retention-web")
    pipeline = RetentionPipeline(
        storage_backend=LocalFilesystemBackend(storage_dir),
        issuer_key=(payload.get("issuer_key") or "web-issuer-key").encode(),
        validator_key=(payload.get("validator_key") or "web-validator-key").encode(),
        challenger_key=(payload.get("challenger_key") or "web-challenger-key").encode(),
    )
    result = pipeline.run_full(
        artifact_id=artifact_id,
        artifact_bytes=artifact_bytes,
        profile_name=payload.get("profile_name", "DSA-CI-Lite"),
        issuer_id=payload.get("issuer_id", "web-ui"),
        validator_id=payload.get("validator_id", "web-validator"),
        min_retention_days=int(payload.get("min_retention_days", 90)),
        max_artifact_bytes=int(payload.get("max_artifact_bytes", 10_000_000)),
        run_possession_challenge=bool(payload.get("run_possession_challenge", True)),
        verify_chain=bool(payload.get("verify_chain", True)),
    )
    return result.to_dict()


@app.post("/v210/retention/challenge")
def retention_challenge(payload: dict) -> dict:
    """Issue a possession challenge and verify the response."""
    from .challenge import issue_possession_challenge, respond_to_challenge, verify_possession_challenge

    artifact_id = payload.get("artifact_id", "test-artifact")
    content = payload.get("content", "")
    artifact_bytes = content.encode("utf-8") if content else b""
    if not artifact_bytes:
        raise HTTPException(status_code=400, detail="content is required")

    challenger_key = (payload.get("challenger_key") or "web-challenger-key").encode()
    responder_key = (payload.get("responder_key") or "web-responder-key").encode()

    challenge = issue_possession_challenge(
        artifact_id=artifact_id,
        artifact_bytes=artifact_bytes,
        challenger_key=challenger_key,
    )
    response = respond_to_challenge(
        challenge=challenge,
        artifact_bytes=artifact_bytes,
        responder_id=payload.get("responder_id", "web-responder"),
        responder_key=responder_key,
    )
    verify_result = verify_possession_challenge(
        challenge=challenge,
        response=response,
        challenger_key=challenger_key,
    )

    return {
        "challenge_id": challenge.challenge_id,
        "artifact_id": artifact_id,
        "byte_range": [challenge.start_offset, challenge.end_offset],
        "artifact_size": len(artifact_bytes),
        "nonce": challenge.nonce,
        "expires_at": challenge.expires_at,
        "verification": verify_result,
    }


# ======================================================================
# v2.10 — Evidence Profile & Budget API
# ======================================================================

@app.get("/v210/evidence/profiles")
def evidence_profiles() -> list[dict]:
    """List all available evidence profiles."""
    from .evidence_profile import PROFILES
    return [
        {
            "profile_id": p.profile_id.value,
            "ceiling_bytes_per_token": p.ceiling_bytes_per_token,
            "base_k": p.base_k,
            "max_adaptive_k": p.max_adaptive_k,
            "ci_modes": list(p.ci_modes),
            "verifier_time_seconds": p.verifier_time_seconds,
            "verifier_peak_mib": p.verifier_peak_mib,
            "default_stochastic_rate_ppm": p.default_stochastic_rate_ppm,
            "v33_header_fields": {
                "tokenizer_id": p.v33_header_fields.tokenizer_id,
                "signal_schema_id": p.v33_header_fields.signal_schema_id,
                "quantization_id": p.v33_header_fields.quantization_id,
                "verifier_work_profile_id": p.v33_header_fields.verifier_work_profile_id,
                "runtime_calibration_id": p.v33_header_fields.runtime_calibration_id,
                "retention_policy_id": p.v33_header_fields.retention_policy_id,
            },
        }
        for p in PROFILES.values()
    ]


@app.post("/v210/evidence/budget")
def evidence_budget(payload: dict) -> dict:
    """Compute evidence budget summary for a profile and token count."""
    from .evidence_profile import get_evidence_profile

    profile_name = payload.get("profile", "DSA-CI-Lite")
    token_count = int(payload.get("token_count", 1000))

    try:
        profile = get_evidence_profile(profile_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    base_k = profile.base_k
    min_bytes_per_token = 6 + base_k * 5  # simplified: _TOKEN + base_k * _TOPK
    total_min_bytes = token_count * min_bytes_per_token
    total_ceiling_bytes = token_count * profile.ceiling_bytes_per_token
    retention_floor_margin = total_ceiling_bytes - total_min_bytes

    return {
        "profile_id": profile.profile_id.value,
        "token_count": token_count,
        "base_k": base_k,
        "max_adaptive_k": profile.max_adaptive_k,
        "ceiling_bytes_per_token": profile.ceiling_bytes_per_token,
        "minimum_reconstructable_bytes_per_token": profile.minimum_reconstructable_bytes_per_token(),
        "estimated_min_total_bytes": total_min_bytes,
        "estimated_ceiling_total_bytes": total_ceiling_bytes,
        "retention_floor_margin_bytes": retention_floor_margin,
        "verifier_time_budget_seconds": profile.verifier_time_seconds,
        "verifier_peak_mib": profile.verifier_peak_mib,
        "ci_modes": list(profile.ci_modes),
    }


# ======================================================================
# v2.10 — Portable Verifier API
# ======================================================================

@app.post("/v210/verifier/verify")
def verifier_verify(payload: dict) -> dict:
    """Run portable verification on compact evidence (simulated)."""
    from .verifier import VerifierWorkCertificate, canonical_serialize_certificate

    # Build a simulated work certificate for demo
    token_count = int(payload.get("token_count", 100))
    profile = payload.get("profile", "DSA-CI-Lite")
    base_k = int(payload.get("base_k", 3))

    bytes_per_token = 6 + base_k * 5  # simplified
    total_bytes = token_count * bytes_per_token

    cert = VerifierWorkCertificate(
        bytes_read=total_bytes,
        bytes_hashed=total_bytes,
        token_records_decoded=token_count,
        candidate_entries_decoded=token_count * base_k,
        varint_bytes_decoded=token_count * 2,
        chunks_verified=max(1, token_count // 100),
        span_events_indexed=0,
        span_overlay_operations=0,
        allocations=token_count * 3,
        maximum_live_bytes=total_bytes * 2,
        full_artifact_materializations=1,
        normalized_runtime_seconds=token_count / 1000.0,
    )

    cert_hash = hashlib.sha256(canonical_serialize_certificate(cert)).hexdigest()

    return {
        "profile": profile,
        "token_count": token_count,
        "base_k": base_k,
        "certificate": {
            "bytes_read": cert.bytes_read,
            "bytes_hashed": cert.bytes_hashed,
            "token_records_decoded": cert.token_records_decoded,
            "candidate_entries_decoded": cert.candidate_entries_decoded,
            "chunks_verified": cert.chunks_verified,
            "allocations": cert.allocations,
            "maximum_live_bytes": cert.maximum_live_bytes,
            "normalized_runtime_seconds": cert.normalized_runtime_seconds,
            "certificate_hash": cert_hash[:32],
        },
    }


# ======================================================================
# v2.10 — Architecture Overview API
# ======================================================================

@app.get("/v210/architecture")
def architecture_overview() -> dict[str, Any]:
    """Return the v2.10 module architecture overview."""
    return {
        "version": "2.10.0",
        "modules": {
            "compact_evidence": {
                "section": "§4",
                "description": "V3.3 binary codec with keyed replay, canonical hashing, trigger bitmask encoding",
                "key_exports": ["encode_compact_sequence", "decode_compact_sequence", "verify_keyed_replay", "EvidenceManifestV33"],
            },
            "evidence_profile": {
                "section": "§4.1",
                "description": "Four CI profiles (Lite, Standard, Deep, Forensic) with budget, timing, and V3.3 header field IDs",
                "key_exports": ["EvidenceProfile", "get_evidence_profile", "PROFILES"],
            },
            "generator": {
                "section": "§3",
                "description": "Model-level generation with V3.3 compact evidence emission, adaptive k-selection, probe integration",
                "key_exports": ["DualStreamGenerator", "GenerationConfig"],
            },
            "tension_map": {
                "section": "§6.1",
                "description": "Signed YAML-based governance rules for adaptive k-widening with HMAC verification",
                "key_exports": ["TensionMap", "TensionRule"],
            },
            "triggers": {
                "section": "§6.2",
                "description": "ThresholdTrigger, SequenceTrigger, CompositeTrigger with TriggerPipeline orchestration",
                "key_exports": ["ThresholdTrigger", "SequenceTrigger", "CompositeTrigger", "TriggerPipeline"],
            },
            "envelope": {
                "section": "§5.3",
                "description": "Streaming-space envelopes with cumulative SHA-256, interim commitments, HMAC-sealed finalization",
                "key_exports": ["EnvelopeSession", "SealedEnvelope", "verify_envelope"],
            },
            "retention_manager": {
                "section": "§7",
                "description": "End-to-end retention pipeline: requirement → store → validate → receipt → challenge → chain verify",
                "key_exports": ["RetentionPipeline", "RetentionPipelineResult", "PipelineStep"],
            },
            "retention_lifecycle": {
                "section": "§7.1",
                "description": "Retention requirement and receipt issuance, HMAC-signed with chain verification",
                "key_exports": ["issue_retention_requirement", "issue_retention_receipt", "verify_receipt_chain"],
            },
            "challenge": {
                "section": "§7.2",
                "description": "Random byte-range possession challenges with HMAC-signed challenges and responses",
                "key_exports": ["issue_possession_challenge", "respond_to_challenge", "verify_possession_challenge"],
            },
            "storage_validator": {
                "section": "§7.3",
                "description": "StorageBackend ABC with LocalFilesystemBackend, hash+size validation",
                "key_exports": ["StorageBackend", "LocalFilesystemBackend", "validate_persisted_artifact"],
            },
            "migration": {
                "section": "§7.4",
                "description": "6 transform types (replicate, compress, encrypt, reformat, shard, archive) with retention floor enforcement",
                "key_exports": ["AllowedTransform", "verify_after_transform"],
            },
            "verifier": {
                "section": "§8",
                "description": "Portable verifier with RetryPolicy, VerifierWorkCertificate, INCONCLUSIVE_INFRA retry, budget enforcement",
                "key_exports": ["verify_with_retry", "VerifierWorkCertificate", "RetryPolicy"],
            },
            "integrity": {
                "section": "§9",
                "description": "CRC32, running SHA-256, canonical JSON serialization, deterministic hashing",
                "key_exports": ["compute_crc32", "compute_running_hash", "canonical_json"],
            },
            "vocab": {
                "section": "§2",
                "description": "AST signal code constants, vocabulary mappings, schema identifiers",
                "key_exports": ["AST_SYCOPHANCY", "AST_RETENTION_FLOOR_VIOLATION", "SIGNAL_SCHEMA_ID_AST1_V210"],
            },
        },
    }
