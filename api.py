from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .service import DualStreamService

app = FastAPI(title="DualStream Browser API", version="0.2.0")
service = DualStreamService()
WEB_ROOT = Path(__file__).resolve().parent / "web"

if WEB_ROOT.exists():
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui/status")
def ui_status() -> dict[str, bool]:
    return {"offline_default": True}


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
