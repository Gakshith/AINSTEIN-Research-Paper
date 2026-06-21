"""Background runner for the live single-paper demo.

Wraps the existing ``AINSTEINController`` (main.py) in a worker thread, exposes
staged progress for polling, and is guarded by ``require_hf_token``. Concurrency is
limited to one job at a time (the pipeline mutates shared artifacts on disk).
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.AINSTEIN.utils.common import hf_token_available

# Ordered human-readable stages surfaced to the UI.
STAGE_SEQUENCE = [
    "Data Ingestion stage",
    "Data Validation stage",
    "Abstract Generalizer stage",
    "Solution stage",
    "Internal Critique stage",
    "External Critique stage",
    "Evaluation stage",
]

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def token_configured() -> bool:
    return hf_token_available()


def active_job_id() -> str | None:
    with _lock:
        for job_id, job in _jobs.items():
            if job["status"] in ("pending", "running"):
                return job_id
    return None


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def start_run(row_index: int | None = None, paper_id: str | None = None) -> dict:
    """Start a live single-paper run. Returns the new job dict.

    Raises RuntimeError if no HF token is configured or a run is already active.
    """
    if not hf_token_available():
        raise RuntimeError("no_token")
    if active_job_id() is not None:
        raise RuntimeError("busy")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "status": "pending",
        "row_index": row_index,
        "paper_id": paper_id,
        "stages": [{"name": s, "state": "pending"} for s in STAGE_SEQUENCE],
        "result": None,
        "error": None,
        "created_at": time.time(),
    }
    with _lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_worker, args=(job_id, row_index, paper_id), daemon=True)
    thread.start()
    return dict(job)


def _set_stage(job_id: str, stage_name: str, state: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for stage in job["stages"]:
            if stage["name"] == stage_name:
                stage["state"] = state


def _worker(job_id: str, row_index: int | None, paper_id: str | None) -> None:
    # Import here to avoid pulling the heavy pipeline import chain at app startup.
    from main import AINSTEINController

    prev_env = {
        "AINSTEIN_ROW_INDEX": os.environ.get("AINSTEIN_ROW_INDEX"),
        "AINSTEIN_PAPER_ID": os.environ.get("AINSTEIN_PAPER_ID"),
    }
    if row_index is not None:
        os.environ["AINSTEIN_ROW_INDEX"] = str(row_index)
    if paper_id:
        os.environ["AINSTEIN_PAPER_ID"] = str(paper_id)

    with _lock:
        _jobs[job_id]["status"] = "running"

    controller = AINSTEINController()
    original_run_stage = controller._run_stage

    def traced_run_stage(stage_name: str, stage_obj):
        _set_stage(job_id, stage_name, "running")
        try:
            result = original_run_stage(stage_name, stage_obj)
        except Exception:
            _set_stage(job_id, stage_name, "failed")
            raise
        _set_stage(job_id, stage_name, "done")
        return result

    controller._run_stage = traced_run_stage  # type: ignore[method-assign]

    try:
        p_final, z_final = controller.run()
        with _lock:
            job = _jobs[job_id]
            job["status"] = "completed"
            job["result"] = {
                "problem_statement": _to_text(p_final),
                "solution": _to_text(z_final),
                "critique_passed": controller.critique_passed,
                "evaluation": _read_evaluation_output(),
            }
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        with _lock:
            job = _jobs[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
    finally:
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    research_solution = getattr(value, "research_solution", None)
    if research_solution is not None:
        return str(research_solution)
    return str(value)


def _read_evaluation_output() -> str | None:
    """Best-effort read of the human-readable evaluation summary, if produced."""
    try:
        from web.services import eval_config

        path = Path(eval_config().output_file)
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None
    return None
