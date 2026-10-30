"""Nexus Agent — Standalone Flask app.

Run:
    pip install -r requirements.txt
    python app.py

Then open http://localhost:5050
"""

import hashlib
import json
import os
import tempfile
import time
from functools import wraps

from flask import (
    Flask, Blueprint, Response, jsonify, render_template, request,
    make_response, stream_with_context, session, redirect, url_for,
)
from werkzeug.utils import secure_filename
import yaml

import config

app = Flask(__name__)
app.secret_key = config.SECRET_KEY


# ═══════════════════════════════════════════════════════════════════
# Auth — simple session-based login
# ═══════════════════════════════════════════════════════════════════

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if config.AUTH_USERS.get(username) == password:
            session["user"] = username
            return redirect(url_for("agent_page"))
        error = "Invalid username or password"
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nexus Agent — Login</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:#0a0b10;color:#e4e8f1;display:flex;align-items:center;justify-content:center;min-height:100vh}}
.card{{background:rgba(17,19,28,.92);border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:40px;width:380px;box-shadow:0 8px 40px rgba(0,0,0,.4)}}
h1{{font-size:1.4em;font-weight:800;margin-bottom:6px;background:linear-gradient(135deg,#6c8cff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.sub{{font-size:.82em;color:#4d5468;margin-bottom:28px}}
label{{display:block;font-size:.75em;font-weight:600;color:#8890a6;margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em}}
input{{width:100%;padding:10px 14px;border-radius:10px;border:1.5px solid rgba(255,255,255,.1);background:rgba(20,22,32,.8);color:#e4e8f1;font-family:inherit;font-size:.88em;margin-bottom:16px;outline:none;transition:border .15s}}
input:focus{{border-color:#6c8cff}}
button{{width:100%;padding:12px;border-radius:10px;border:none;background:linear-gradient(135deg,#6c8cff,#a78bfa);color:#fff;font-weight:700;font-size:.88em;cursor:pointer;font-family:inherit;transition:opacity .15s}}
button:hover{{opacity:.9}}
.err{{color:#f87171;font-size:.8em;margin-bottom:12px;text-align:center}}
</style>
</head><body>
<div class="card">
<h1>Nexus Agent</h1>
<div class="sub">MMM Report Platform</div>
{'<div class="err">'+error+'</div>' if error else ''}
<form method="POST">
<label>Username</label><input name="username" autofocus>
<label>Password</label><input name="password" type="password">
<button type="submit">Sign In</button>
</form>
</div>
</body></html>"""


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ═══════════════════════════════════════════════════════════════════
# Agent UI page
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
@auth_required
def agent_page():
    user = session.get("user", "agent")
    return render_template("agent.html", user_email=f"{user}@nexus.local")


# ═══════════════════════════════════════════════════════════════════
# Agent chat endpoints
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/agent/send", methods=["POST"])
@auth_required
def agent_send():
    from services.agent_service import MMMAgentService
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "message is required"}), 400
    history = data.get("history", [])
    sid = data.get("session_id")
    result = MMMAgentService.chat(message, history, session_id=sid)
    return jsonify(result)


@app.route("/api/agent/stream", methods=["POST"])
@auth_required
def agent_stream():
    from services.agent_service import MMMAgentService
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"success": False, "error": "message is required"}), 400
    history = data.get("history", [])
    session_id = data.get("session_id")

    def generate():
        result = MMMAgentService.chat(message, history, session_id=session_id)
        if result.get("tool_calls"):
            yield f"data: {json.dumps({'type': 'tools', 'tools': result['tool_calls']})}\n\n"
        for block in result.get("blocks", []):
            if block["type"] == "report":
                yield f"data: {json.dumps({'type': 'report', 'report_id': block['report_id'], 'report_type': block['report_type'], 'size_bytes': block['size_bytes'], 'preview_url': block['preview_url'], 'download_url': block['download_url']})}\n\n"
            elif block["type"] == "job":
                yield f"data: {json.dumps({'type': 'job', 'job_id': block['job_id'], 'label': block['label'], 'report_type': block['report_type']})}\n\n"
            elif block["type"] == "code":
                yield f"data: {json.dumps({'type': 'code', 'content': block['content'], 'language': block.get('language', 'text')})}\n\n"
            elif block["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': block['content']})}\n\n"
            else:
                text = block.get("content", "")
                lines = text.split("\n")
                for i, line in enumerate(lines):
                    words = line.split(" ")
                    for j, word in enumerate(words):
                        chunk = word + (" " if j < len(words) - 1 else "")
                        if chunk:
                            yield f"data: {json.dumps({'type': 'word', 'content': chunk})}\n\n"
                            time.sleep(0.02)
                    if i < len(lines) - 1:
                        yield f"data: {json.dumps({'type': 'word', 'content': chr(10)})}\n\n"
                yield f"data: {json.dumps({'type': 'text_end'})}\n\n"
        if result.get("sources"):
            yield f"data: {json.dumps({'type': 'sources', 'sources': result['sources']})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'raw_content': result.get('raw_content', '')})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════
# File upload
# ═══════════════════════════════════════════════════════════════════

TABULAR_EXTS = (".csv", ".rds")
TEXT_EXTS = (".yaml", ".yml", ".py", ".html", ".htm", ".qmd", ".md", ".txt", ".json")
MAX_TEXT_UPLOAD_BYTES = 5 * 1024 * 1024


@app.route("/api/agent/upload", methods=["POST"])
@auth_required
def agent_upload():
    from services.agent_service import _UPLOADED_FILES, UPLOADS_DIR
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400
    name = f.filename.lower()
    if name.endswith(TEXT_EXTS):
        return _save_text_upload(f, _UPLOADED_FILES, UPLOADS_DIR)
    if not name.endswith(TABULAR_EXTS):
        return jsonify({"success": False, "error": "Supported: " + ", ".join(TABULAR_EXTS + TEXT_EXTS)}), 400
    try:
        import pandas as pd
        if name.endswith(".csv"):
            df = pd.read_csv(f.stream)
        else:
            import pyreadr
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".rds")
            try:
                f.save(tmp.name)
                result = pyreadr.read_r(tmp.name)
                df = list(result.values())[0]
            finally:
                os.unlink(tmp.name)
        records = df.head(500).to_dict(orient="records")
        file_id = hashlib.sha256(f.filename.encode() + str(len(records)).encode()).hexdigest()[:10]
        _UPLOADED_FILES[file_id] = {
            "filename": f.filename, "columns": list(df.columns),
            "rows": len(df), "preview_rows": len(records), "data": records,
        }
        return jsonify({"success": True, "file_id": file_id, "filename": f.filename,
                         "columns": list(df.columns), "rows": len(df),
                         "preview": [{ k: str(v)[:50] for k,v in r.items()} for r in records[:3]]})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to parse: {e}"}), 500


def _save_text_upload(f, registry: dict, uploads_dir):
    raw = f.read(MAX_TEXT_UPLOAD_BYTES + 1)
    if len(raw) > MAX_TEXT_UPLOAD_BYTES:
        return jsonify({"success": False, "error": "File too large (max 5 MB)"}), 400
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"success": False, "error": "File is not UTF-8 text"}), 400
    safe_name = secure_filename(f.filename) or "upload.txt"
    is_yaml = safe_name.lower().endswith((".yaml", ".yml"))
    if is_yaml:
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as e:
            return jsonify({"success": False, "error": f"Invalid YAML: {e}"}), 400
        if not isinstance(parsed, dict):
            return jsonify({"success": False, "error": "YAML must be a mapping at the top level"}), 400
    file_id = hashlib.sha256(raw + safe_name.encode()).hexdigest()[:10]
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / f"{file_id}_{safe_name}"
    path.write_bytes(raw)
    registry[file_id] = {"kind": "text", "filename": f.filename, "path": str(path)}
    lines = text.count("\n") + 1
    return jsonify({"success": True, "file_id": file_id, "filename": f.filename,
                    "kind": "config" if is_yaml else "text", "lines": lines})


@app.route("/api/agent/upload/<file_id>")
@auth_required
def agent_get_upload(file_id):
    from services.agent_service import _UPLOADED_FILES
    entry = _UPLOADED_FILES.get(file_id)
    if not entry:
        return jsonify({"success": False, "error": "File not found"}), 404
    return jsonify({"success": True, **entry})


# ═══════════════════════════════════════════════════════════════════
# Report preview / download
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/agent/jobs/<job_id>")
@auth_required
def agent_job_status(job_id):
    from services.report_jobs import get_job
    job = get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found (the server may have restarted)"}), 404
    return jsonify({"success": True, **job})


@app.route("/api/agent/preview/<report_id>")
@auth_required
def agent_preview(report_id):
    from services.agent_service import _PREVIEW_CACHE
    html = _PREVIEW_CACHE.get(report_id)
    if not html:
        return jsonify({"error": "Report not found or expired"}), 404
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


@app.route("/api/agent/download/<report_id>")
@auth_required
def agent_download(report_id):
    from services.agent_service import _PREVIEW_CACHE
    html = _PREVIEW_CACHE.get(report_id)
    if not html:
        return jsonify({"error": "Report not found or expired"}), 404
    resp = make_response(html)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    resp.headers["Content-Disposition"] = f"attachment; filename=report_{report_id}.html"
    return resp


# ═══════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Nexus Agent running at http://localhost:5050\n")
    app.run(debug=True, port=5050, host="0.0.0.0")
