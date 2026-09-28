from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from dualstream.service import DualStreamService


class DummyTask:
    task_id = "task1"


class DummyResult:
    task_id = "task1"
    attempts = []
    candidates = []
    frames = []
    findings = []
    audit_summary = {"num_frames": 0}
    metrics = {"coherence_score": 1.0}


def test_start_arc_solve_task(monkeypatch, tmp_path) -> None:
    service = DualStreamService()

    monkeypatch.setattr("dualstream.service.load_task", lambda _: DummyTask())

    class DummySolver:
        def __init__(self, *_args, **_kwargs):
            pass

        def solve_task(self, _task):
            return DummyResult()

    monkeypatch.setattr("dualstream.service.ArcSolver", DummySolver)

    def fake_write_task_artifacts(_result, outdir, include_rankings=True):
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "predictions.json").write_text("[]", encoding="utf-8")
        (outdir / "summary_metrics.json").write_text('{"coherence_score": 1.0}', encoding="utf-8")
        (outdir / "audit.json").write_text('{"summary": {"num_frames": 0}}', encoding="utf-8")
        (outdir / "trace.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr("dualstream.service.write_task_artifacts", fake_write_task_artifacts)

    def fake_write_submission(_results, output):
        output.write_text(json.dumps({"ok": True}), encoding="utf-8")

    monkeypatch.setattr("dualstream.service.write_submission", fake_write_submission)

    task_path = tmp_path / "task.json"
    task_path.write_text("{}", encoding="utf-8")
    job = service.start_arc_solve_task({"task": str(task_path), "outdir": str(tmp_path)})

    # wait for completion
    for _ in range(200):
        state = service.get_job(job.id)
        assert state is not None
        if state.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    final = service.get_job(job.id)
    assert final is not None
    assert final.status == "completed"
    assert final.result is not None
    assert "metrics" in final.result


def test_list_scripts_exposes_repo_scripts() -> None:
    service = DualStreamService()
    scripts = service.list_scripts()
    assert any(item["name"] == "download_model.py" for item in scripts)


def test_preflight_script_rejects_unknown_script(tmp_path) -> None:
    service = DualStreamService()
    result = service.preflight_script({"script_name": "missing.py", "outdir": str(tmp_path)})
    assert result["ok"] is False
    assert any("Script not found" in err for err in result["errors"])


def test_writable_directory_probe_preserves_existing_file(tmp_path):
    existing = tmp_path / ".dualstream_write_test"
    existing.write_text("user data")
    assert DualStreamService._validate_writable_dir(tmp_path, "Output") == []
    assert existing.read_text() == "user data"
    assert list(tmp_path.iterdir()) == [existing]


def test_cancelled_queued_job_never_runs():
    service = DualStreamService()
    service._executor.shutdown(wait=True)
    service._executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    started = threading.Event()
    called = []

    def blocking(job, cancelled):
        started.set()
        assert release.wait(5)
        return {}

    try:
        first = service.create_job("blocker", blocking)
        assert started.wait(5)
        queued = service.create_job("queued", lambda *args: called.append(True) or {})
        assert service.cancel_job(queued.id)
    finally:
        release.set()
        service._executor.shutdown(wait=True)
    assert not called
    assert service.get_job(queued.id).status == "cancelled"
    assert service.get_job(queued.id).ended_at is not None
    assert service.cancel_job(first.id) is False
