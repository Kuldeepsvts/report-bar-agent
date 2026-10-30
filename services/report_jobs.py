"""report_jobs.py — Report HTML cache + background runner for long (real-S3) builds."""

import hashlib
import subprocess
import sys
import threading
import time
import uuid

from services.repo_tools import EXTERNAL_REPO_DIR

JOB_TIMEOUT_S = 20 * 60

PREVIEW_CACHE: dict[str, str] = {}
_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def cache_report(html: str) -> str:
    report_id = hashlib.sha256((html + str(time.time())).encode()).hexdigest()[:12]
    PREVIEW_CACHE[report_id] = html
    return report_id


def run_build(code: str, timeout: int) -> tuple[str, str]:
    """Run report-building code in a subprocess at the repo root. Returns (html, error)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(EXTERNAL_REPO_DIR),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", f"Timed out after {timeout}s"
    if result.returncode != 0:
        err = result.stderr.strip()
        return "", err[-6000:] if len(err) > 6000 else err
    if not result.stdout.strip():
        return "", f"Empty output. stderr: {result.stderr.strip()[-2000:]}"
    return result.stdout, ""


def start_job(code: str, report_type: str, label: str) -> str:
    job_id = uuid.uuid4().hex[:10]
    job = {
        "job_id": job_id, "report_type": report_type, "label": label,
        "status": "running", "started": time.time(), "finished": None,
        "error": "", "report_id": "", "size_bytes": 0,
    }
    with _LOCK:
        _JOBS[job_id] = job

    def _worker():
        html, err = run_build(code, JOB_TIMEOUT_S)
        with _LOCK:
            job["finished"] = time.time()
            if err:
                job["status"], job["error"] = "failed", err
            else:
                job["report_id"] = cache_report(html)
                job["size_bytes"] = len(html)
                job["status"] = "done"

    threading.Thread(target=_worker, daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        out = dict(job)
    end = out["finished"] or time.time()
    out["elapsed_s"] = round(end - out["started"], 1)
    if out["status"] == "done":
        out["preview_url"] = f"/api/agent/preview/{out['report_id']}"
        out["download_url"] = f"/api/agent/download/{out['report_id']}"
    return out
