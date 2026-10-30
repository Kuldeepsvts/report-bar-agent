"""mmm_agent_service.py — Code-maintenance agent for the unified mmm_report_platform package.

Uses LangChain + LangGraph with Cortex (OpenAI-compatible gateway):
  - LangGraph tool-calling agent with file/git tools for EliLillyCo/report.git,
    which contains the mmm_report_platform package (demo reports + DTC/HCP
    impact model reports, each report type in its own reports/<type>/ folder)
  - FAISS RAG over the mmm_report_platform source code for context
  - Multi-turn memory via MemorySaver checkpointer
  - Token refresh per-request to handle SSO expiry
  - Automatic retry on transient LLM failures
"""

import json as _json
import time
import uuid
from pathlib import Path
from typing import Any, Annotated, Literal, Optional

import yaml
from langchain_openai import ChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from typing_extensions import TypedDict

import config
from services import report_jobs
from services.repo_tools import (
    execute_tool as _exec_repo_tool,
    EXTERNAL_REPO_DIR,
)

_PREVIEW_CACHE = report_jobs.PREVIEW_CACHE
_UPLOADED_FILES: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Cortex connection — token refreshed per-request
# ═══════════════════════════════════════════════════════════════════════════════

OPENAI_API_BASE = config.CORTEX_API_BASE
LLM_MODEL = config.LLM_MODEL
EMBEDDING_MODEL = config.EMBEDDING_MODEL

_TOKEN_CACHE = {"token": None, "expires": 0}


def _get_cortex_token() -> str:
    now = time.time()
    if _TOKEN_CACHE["token"] and now < _TOKEN_CACHE["expires"]:
        return _TOKEN_CACHE["token"]
    from light_client import LIGHTClient
    client = LIGHTClient()
    token = client.get_auth_header()["Authorization"].split(" ")[1]
    _TOKEN_CACHE["token"] = token
    _TOKEN_CACHE["expires"] = now + 240
    return token


def _get_llm() -> ChatOpenAI:
    token = _get_cortex_token()
    return ChatOpenAI(
        model_name=LLM_MODEL,
        base_url=OPENAI_API_BASE,
        api_key=token,
        temperature=0,
        request_timeout=120,
    )


def _get_embeddings() -> AzureOpenAIEmbeddings:
    token = _get_cortex_token()
    return AzureOpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        azure_endpoint=OPENAI_API_BASE,
        api_version="dummy_api_version",
        api_key=token,
        check_embedding_ctx_length=False,
        timeout=60.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LangChain Tools — all scoped to github.com/EliLillyCo/report.git
# ═══════════════════════════════════════════════════════════════════════════════

_LAST_GENERATED_REPORTS: list[dict] = []

@tool
def clone_or_update() -> str:
    """Clone the mmm_report_platform repo if not present locally, else pull latest. Call this first before any other tool."""
    r = _exec_repo_tool("repo_clone_or_update", {})
    return r.get("data", r.get("error", "unknown error"))


@tool
def list_files(pattern: str = "") -> str:
    """List files in the mmm_report_platform repo. Optional glob pattern e.g. '**/*.py'."""
    r = _exec_repo_tool("repo_list_files", {"pattern": pattern})
    return r.get("data", r.get("error", "unknown error"))


@tool
def read_file(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a file from the mmm_report_platform repo. Returns content with line numbers.
    For large files, pass start_line/end_line (1-based, inclusive) to read just a range."""
    r = _exec_repo_tool("repo_read_file", {"path": path, "start_line": start_line, "end_line": end_line})
    return r.get("data", r.get("error", "unknown error"))


@tool
def write_file(path: str, content: str) -> str:
    """Create a NEW file (or fully rewrite a small one) in the repo. Creates parent dirs.
    For changing part of an existing file, use edit_file instead — it is safer and cheaper."""
    r = _exec_repo_tool("repo_write_file", {"path": path, "content": content})
    return r.get("data", r.get("error", "unknown error"))


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace one exact block of text in an existing repo file.

    old_text must match the file EXACTLY once (copy it from read_file output WITHOUT the
    'NNNN | ' line-number prefix, keeping indentation). Include 2-3 surrounding lines if the
    snippet alone is not unique. Prefer this over write_file for every edit to existing code.
    """
    r = _exec_repo_tool("repo_edit_file", {"path": path, "old_text": old_text, "new_text": new_text})
    return r.get("data", r.get("error", "unknown error"))


@tool
def search_code(pattern: str, file_pattern: str = "") -> str:
    """Search for a text/regex pattern across files in the mmm_report_platform repo."""
    r = _exec_repo_tool("repo_search_code", {"pattern": pattern, "file_pattern": file_pattern})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_status() -> str:
    """Show git status of the mmm_report_platform repo."""
    r = _exec_repo_tool("repo_status", {})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_diff(path: str = "") -> str:
    """Show git diff in the mmm_report_platform repo. Optional: pass a file path."""
    r = _exec_repo_tool("repo_diff", {"path": path} if path else {})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_branch(name: str) -> str:
    """Create and switch to a new branch in the mmm_report_platform repo."""
    r = _exec_repo_tool("repo_branch", {"name": name})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_checkout(name: str) -> str:
    """Switch to an existing branch in the mmm_report_platform repo."""
    r = _exec_repo_tool("repo_checkout", {"name": name})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_commit(message: str) -> str:
    """Commit changes under mmm_report_platform/, README.md, pyproject.toml and tests/.
    Other untracked files are NOT committed; they are listed in the result."""
    r = _exec_repo_tool("repo_commit", {"message": message})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_push() -> str:
    """Push the current branch to origin."""
    r = _exec_repo_tool("repo_push", {})
    return r.get("data", r.get("error", "unknown error"))


WORKSPACE_DIR = Path(__file__).resolve().parent.parent / ".workspace"
CONFIGS_DIR = WORKSPACE_DIR / "configs"
UPLOADS_DIR = WORKSPACE_DIR / "uploads"
IMPACT_MODEL_TYPES = ("dtc_impact_model", "hcp_impact_model")

_LAST_STARTED_JOBS: list[dict] = []


def _ensure_repo() -> str:
    if not (EXTERNAL_REPO_DIR / ".git").exists():
        r = _exec_repo_tool("repo_clone_or_update", {})
        if not r.get("success"):
            return f"Failed to clone repo: {r.get('error')}"
    return ""


def _run_report_subprocess(code: str, timeout: int = 30) -> tuple[str, str]:
    """Run code in a subprocess at the repo root. Returns (stdout, error)."""
    err = _ensure_repo()
    if err:
        return "", err
    return report_jobs.run_build(code, timeout)


def _cache_report(html: str, report_type: str) -> str:
    report_id = report_jobs.cache_report(html)
    _LAST_GENERATED_REPORTS.append({
        "report_id": report_id,
        "report_type": report_type,
        "size_bytes": len(html),
    })
    return report_id


def _resolve_config_path(config_path: str) -> tuple[str, str]:
    """Return (path_for_subprocess, error). Accepts repo-relative paths or files in .workspace/."""
    p = Path(config_path).expanduser()
    if p.is_absolute():
        resolved = p.resolve()
        if not str(resolved).startswith(str(WORKSPACE_DIR.resolve())):
            return "", (f"config_path {config_path!r} is outside the repo and the agent workspace. "
                        f"Ask the user to upload the file (paperclip) or paste it so you can save_config() it.")
        if not resolved.exists():
            return "", f"Config not found: {config_path}"
        return str(resolved), ""
    rel = config_path.lstrip("/")
    for candidate in (rel, f"mmm_report_platform/{rel}", f"mmm_report_platform/config/{rel}"):
        if (EXTERNAL_REPO_DIR / candidate).is_file():
            return candidate, ""
    return "", (f"Config not found: {config_path!r}. Use list_report_configs() for bundled configs, or the "
                f"path returned by save_config()/get_uploaded_file() for user-provided ones.")


@tool
def list_report_types() -> str:
    """List all available report types that can be generated.

    Returns both demo report types (executive, sales, product, demographic,
    channel — no config_path needed) and impact-model report types
    (dtc_impact_model, hcp_impact_model — these REQUIRE config_path).
    """
    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from mmm_report_platform import list_report_types; "
        "r = list_report_types(); "
        "[print(f'  {n:18s} — {d}') for n, d in r.items()]; "
        "print(f'{len(r)} report types available')"
    )
    out, err = _run_report_subprocess(code)
    if err:
        return f"Error: {err}"
    return out


@tool
def list_report_configs(report_type: Literal["dtc_impact_model", "hcp_impact_model"]) -> str:
    """List YAML configs available for an impact-model report type: the ones bundled in the
    repo (mmm_report_platform/config/) AND any the user provided (.workspace/configs/ and uploads).

    Use this BEFORE generate_report() for dtc_impact_model/hcp_impact_model if you don't
    already have a config_path. There is no fixed brand list.
    """
    code = (
        "import sys; sys.path.insert(0, '.'); "
        f"from mmm_report_platform.reports.{report_type}.build import list_configs; "
        "r = list_configs(); "
        "[print(f'  mmm_report_platform/config/{n:40s} — {d}') for n, d in r.items()]; "
        "print(f'{len(r)} bundled config(s)')"
    )
    out, err = _run_report_subprocess(code)
    if err:
        return f"Error: {err}"
    user_cfgs = [p for d in (CONFIGS_DIR, UPLOADS_DIR) if d.exists() for p in sorted(d.glob("*.y*ml"))]
    if user_cfgs:
        out += "\nUser-provided configs (pass the full path as config_path):\n"
        out += "\n".join(f"  {p}" for p in user_cfgs)
    return out


def _check_config_schema(path: str) -> str:
    code = (
        "import sys, yaml; sys.path.insert(0, '.'); "
        "from mmm_report_platform.reports.dtc_impact_model.load_data import _validate_config_schema; "
        f"p = {path!r}; _validate_config_schema(yaml.safe_load(open(p)), p); print('OK')"
    )
    _, err = _run_report_subprocess(code)
    if not err:
        return ""
    lines = [l for l in err.strip().splitlines() if l.strip()]
    msg = next((l for l in lines if "missing required field" in l), lines[-1] if lines else err)
    return msg.split("ValueError: ", 1)[-1]


@tool
def save_config(filename: str, yaml_text: str) -> str:
    """Save a YAML config the user pasted in chat to the agent workspace (NOT the repo) and
    return its path for generate_report(config_path=...).

    Validates YAML syntax and the required impact-model fields. Only if the user asks to add
    it to the repo permanently: write_file it under mmm_report_platform/config/ on a branch + PR.
    """
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return f"Error: not valid YAML — {e}"
    if not isinstance(parsed, dict):
        return "Error: YAML must be a mapping at the top level (brand:, quarter:, base_dir:, ...)."
    name = Path(filename).name
    if not name.lower().endswith((".yaml", ".yml")):
        name += ".yaml"
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    dest = CONFIGS_DIR / name
    dest.write_text(yaml_text, encoding="utf-8")
    problems = _check_config_schema(str(dest))
    note = f"\nSchema problem (a real-data build will fail until fixed): {problems}" if problems else "\nSchema check: OK"
    return f"Saved to {dest}{note}"


@tool
def generate_report(
    report_type: str = "executive",
    title: str = "",
    config_path: str = "",
    theme: Literal["dark", "light", "slate", "studio"] = "dark",
    mode: Literal["mock", "real"] = "mock",
) -> str:
    """Generate an HTML report from the LOCAL repo code and show it in the UI preview.

    Runs the locally edited code in external_repos/report/mmm_report_platform/ (not a
    pip-installed version), so edits show up immediately.

    Demo types (executive, sales, product, demographic, channel):
        generate_report("executive"); generate_report("sales", title="Q3 Sales")

    Impact-model types (dtc_impact_model, hcp_impact_model) — config_path REQUIRED. Brand,
    colors, quarters, stream all come from that YAML. Accepts a repo path
    ("mmm_report_platform/config/DTC_Impact_Model_Zepbound.yaml") or a user-provided config
    path returned by save_config()/get_uploaded_file().
        mode="mock" — synthetic data, ~5-10s, preview appears right away.
        mode="real" — live S3 data. Takes several minutes, so it runs as a BACKGROUND JOB:
                      this returns immediately with a job_id and the user sees a live
                      progress card that turns into the preview when done. Do not wait or
                      poll in a loop — just tell the user it's building.
    theme: "dark", "light", "slate" (calm mid-tone) or "studio" (slate colours + distinctive borderless folder-style boxes) — visual only. title: demo reports only.
    """
    is_impact_model = report_type in IMPACT_MODEL_TYPES

    if is_impact_model:
        if not config_path:
            return (f"Error: {report_type} requires config_path. Call list_report_configs('{report_type}'), "
                    f"or ask the user to upload/paste their config.")
        config_path, err = _resolve_config_path(config_path)
        if err:
            return f"Error: {err}"

    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from mmm_report_platform import build_report; "
        f"html = build_report({report_type!r}"
    )
    if is_impact_model:
        pkg_mode = "s3" if mode == "real" else "mock"
        code += f", config_path={config_path!r}, theme={theme!r}, mode={pkg_mode!r}"
    elif title:
        code += f", title={title!r}"
    code += "); sys.stdout.write(html)"

    if is_impact_model and mode == "real":
        err = _ensure_repo()
        if err:
            return f"Error: {err}"
        label = f"{Path(config_path).stem} · real S3 data"
        job_id = report_jobs.start_job(code, report_type, label)
        _LAST_STARTED_JOBS.append({"job_id": job_id, "label": label, "report_type": report_type})
        return (f"Started background build job {job_id} ({label}). The user now sees a live progress "
                f"card that turns into the preview when finished (usually a few minutes). Tell them that "
                f"and that they can keep chatting; use check_report_job('{job_id}') only if asked later.")

    html, err = _run_report_subprocess(code, timeout=120 if is_impact_model else 30)
    if err:
        return f"Error generating report: {err}"

    _cache_report(html, report_type)
    return (
        f"Successfully generated {report_type} report ({len(html):,} bytes) from local code. "
        f"The preview and download buttons are now visible to the user."
    )


@tool
def check_report_job(job_id: str) -> str:
    """Check the status of a background real-data report build started by generate_report."""
    job = report_jobs.get_job(job_id)
    if not job:
        return f"Unknown job_id {job_id!r} (jobs are lost if the server restarts)."
    if job["status"] == "running":
        return f"Job {job_id} still running ({job['elapsed_s']:.0f}s so far)."
    if job["status"] == "failed":
        return f"Job {job_id} FAILED after {job['elapsed_s']:.0f}s:\n{job['error'][-3000:]}"
    return (f"Job {job_id} done in {job['elapsed_s']:.0f}s ({job['size_bytes']:,} bytes). "
            f"The user's progress card already turned into Preview/Download.")


@tool
def probe_s3_paths(config_path: str) -> str:
    """Check which S3 folders/files a config points at actually exist (model config YAMLs,
    reporting/model_output/input_data folders for current and previous quarter).

    Use when a real-data build fails with a missing-file error, or before the first real build
    of a new config. Takes seconds — does not build the report.
    """
    resolved, err = _resolve_config_path(config_path)
    if err:
        return f"Error: {err}"
    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from mmm_report_platform._cli import _run_probe; "
        f"_run_probe({resolved!r})"
    )
    out, err = _run_report_subprocess(code, timeout=180)
    return out if not err else f"Probe failed: {err}"


@tool
def validate_report_type(report_type: str, config_path: str = "") -> str:
    """Smoke-test a report type end-to-end — REQUIRED before committing a new or changed report.

    Checks it is registered in REPORT_REGISTRY, build() runs (with config_path + mode="mock"
    for config-driven types; defaults otherwise), and the output is a complete HTML document.
    Reports size and build time. On FAIL, fix the code from the traceback and re-run.
    """
    call_args = ""
    if config_path:
        config_path, err = _resolve_config_path(config_path)
        if err:
            return f"Error: {err}"
        call_args = f"config_path={config_path!r}, mode='mock'"
    code = "\n".join([
        "import sys, time; sys.path.insert(0, '.')",
        "from mmm_report_platform.reports import REPORT_REGISTRY",
        f"name = {report_type!r}",
        "if name not in REPORT_REGISTRY:",
        "    sys.exit(f'NOT REGISTERED — add it to REPORT_REGISTRY in mmm_report_platform/reports/__init__.py. Registered: {list(REPORT_REGISTRY)}')",
        "fn, desc = REPORT_REGISTRY[name]",
        "t = time.time()",
        f"html = fn({call_args})",
        "dt = time.time() - t",
        "if not isinstance(html, str) or len(html) < 500:",
        "    sys.exit(f'build() returned {type(html).__name__} ({len(html) if isinstance(html, str) else \"?\"} chars) — expected a full HTML string')",
        "low = html.lower()",
        "if not low.lstrip().startswith(('<!doctype', '<html')):",
        "    sys.exit('output does not start with <!DOCTYPE html> or <html>')",
        "if '</html>' not in low[-1000:]:",
        "    sys.exit('output is missing a closing </html>')",
        "print(f'PASS {name}: {len(html):,} bytes in {dt:.1f}s — {desc}')",
    ])
    out, err = _run_report_subprocess(code, timeout=120)
    return out.strip() if not err else f"FAIL {report_type}:\n{err}"


@tool
def create_pr(title: str, body: str = "") -> str:
    """Create a GitHub Pull Request from the current branch to main.

    IMPORTANT: You must be on a feature branch (not main). The workflow is:
    1. git_branch("my-feature")
    2. edit_file / write_file
    3. git_commit("description of changes")
    4. create_pr("PR title", "Description of what changed and why")

    This pushes the branch and opens the PR on GitHub automatically.
    """
    r = _exec_repo_tool("repo_create_pr", {"title": title, "body": body})
    return r.get("data", r.get("error", "unknown error"))


@tool
def run_script(script: str) -> str:
    """Run a Python script file that already exists inside the repo (e.g. a test file).
    Returns stdout, stderr, and exit status. Timeout: 60 seconds."""
    r = _exec_repo_tool("repo_run_script", {"script": script})
    return r.get("data", r.get("error", "unknown error"))


@tool
def run_python(code: str, timeout: int = 60) -> str:
    """Run an inline Python snippet at the repo root (mmm_report_platform is importable).

    Use it to reproduce bugs, inspect data shapes, or test a function — never write scratch
    scripts into the repo for this. Returns stdout/stderr (traceback tail kept) and exit
    status. timeout max 300s; for real-S3 report builds use generate_report(mode="real").
    """
    r = _exec_repo_tool("repo_run_python", {"code": code, "timeout": timeout})
    return r.get("data", r.get("error", "unknown error"))


@tool
def generate_report_custom(report_type: str, data_json: str, title: str = "") -> str:
    """Generate a DEMO report (executive/sales/product/demographic/channel) using
    custom user-provided data from the LOCAL repo code. Not applicable to
    dtc_impact_model/hcp_impact_model — those are config-driven, not data-driven.

    data_json should be a JSON string representing a dict with any of these keys:
      kpis, regions, countries, products, ethnicities, age_groups, genders, channels

    Any keys not provided will fall back to the standard dataset.
    The UI will automatically show preview and download buttons.
    """
    try:
        custom_data = _json.loads(data_json)
    except _json.JSONDecodeError as e:
        return f"Error parsing JSON: {e}"

    kwargs = {}
    if title:
        kwargs["title"] = title
    kwargs.update(custom_data)
    kwargs_json = _json.dumps(kwargs, default=str)

    code = (
        f"import sys, json; "
        f"sys.path.insert(0, '.'); "
        f"from mmm_report_platform import build_report; "
        f"kw = json.loads({kwargs_json!r}); "
        f"html = build_report({report_type!r}, **kw); "
        f"sys.stdout.write(html)"
    )

    html, err = _run_report_subprocess(code)
    if err:
        return f"Error generating report: {err}"

    _cache_report(html, report_type)
    return (
        f"Successfully generated custom {report_type} report ({len(html):,} bytes) from local code. "
        f"The preview and download buttons are now visible to the user."
    )


@tool
def get_uploaded_file(file_id: str) -> str:
    """Retrieve a file the user attached in the chat (the message contains its file_id).

    - CSV/RDS: returns columns, row count and the first 20 rows (use with generate_report_custom).
    - YAML config: returns the text AND its saved path — pass that path straight to
      generate_report(config_path=...) / probe_s3_paths().
    - .py / .html / .qmd / .md / .txt / .json (starter code, sample report, spec): returns the
      text so you can onboard it as a new report type or follow it as instructions.
    """
    entry = _UPLOADED_FILES.get(file_id)
    if not entry:
        return f"Error: file_id '{file_id}' not found. It may have expired (server restart) — ask the user to re-attach it."
    if entry.get("kind") == "text":
        text = Path(entry["path"]).read_text(encoding="utf-8", errors="replace")
        limit = 60_000
        clipped = text if len(text) <= limit else (
            text[:limit] + f"\n...[{len(text) - limit} chars cut — read the rest with run_python from {entry['path']}]")
        hint = f"\nUse as config_path: {entry['path']}" if entry["path"].lower().endswith((".yaml", ".yml")) else ""
        return f"File: {entry['filename']} ({len(text):,} chars)\nSaved at: {entry['path']}{hint}\n\n{clipped}"
    data_str = _json.dumps(entry["data"][:20], default=str)
    return (
        f"File: {entry['filename']}\n"
        f"Columns: {', '.join(entry['columns'])}\n"
        f"Rows: {entry['rows']}\n"
        f"First 20 rows (JSON):\n{data_str}"
    )


AGENT_TOOLS = [
    clone_or_update, list_files, read_file, write_file, edit_file, search_code,
    git_status, git_diff, git_branch, git_checkout, git_commit, git_push,
    list_report_types, list_report_configs, save_config,
    generate_report, generate_report_custom, check_report_job, probe_s3_paths,
    validate_report_type, create_pr, run_script, run_python, get_uploaded_file,
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAISS RAG knowledge base — index the mmm_report_platform source code
# ═══════════════════════════════════════════════════════════════════════════════

_VECTORSTORE = None
_FAISS_PATH = Path(__file__).parent.parent.parent / ".cache" / "mmm_agent_faiss"


def _build_knowledge_base(force: bool = False):
    global _VECTORSTORE
    if _VECTORSTORE is not None and not force:
        return _VECTORSTORE

    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        return None

    try:
        embeddings = _get_embeddings()
    except Exception:
        return None

    repo_dir = EXTERNAL_REPO_DIR
    if not (repo_dir / ".git").exists():
        return None

    if _FAISS_PATH.exists() and not force:
        try:
            _VECTORSTORE = FAISS.load_local(
                str(_FAISS_PATH), embeddings, allow_dangerous_deserialization=True
            )
            return _VECTORSTORE
        except Exception:
            pass

    docs = []
    for fpath in sorted(repo_dir.rglob("*")):
        if not fpath.is_file():
            continue
        if ".git" in fpath.parts:
            continue
        if fpath.suffix not in (".py", ".js", ".html", ".css", ".yaml", ".yml", ".toml", ".md"):
            continue
        if "__pycache__" in str(fpath) or ".egg-info" in str(fpath):
            continue
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(fpath.relative_to(repo_dir))
        docs.append(Document(page_content=text, metadata={"source": rel}))

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    try:
        _VECTORSTORE = FAISS.from_documents(chunks, embeddings)
        _FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _VECTORSTORE.save_local(str(_FAISS_PATH))
    except Exception as e:
        print(f"[MMM Agent] FAISS build failed: {e}")
        return None

    return _VECTORSTORE


def _retrieve_context(question: str, k: int = 6) -> str:
    vs = _build_knowledge_base()
    if vs is None:
        return ""
    try:
        docs = vs.as_retriever(search_kwargs={"k": k}).invoke(question)
        if not docs:
            return ""
        parts = []
        for d in docs:
            src = d.metadata.get("source", "?")
            parts.append(f"--- {src} ---\n{d.page_content}")
        return "\n\n".join(parts)
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LangGraph Agent — rebuilt per-request with fresh token
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """You are Nexus, a report-platform agent. You maintain ONE git repo
(https://github.com/EliLillyCo/report.git) containing ONE unified Python package: `mmm_report_platform`.
Every report TYPE lives in its own folder under `mmm_report_platform/reports/<type>/`, each exposing
a `build(...)` function. A shared rendering engine (config parsing, D3/HTML assembly, the 6-tab
section library) lives in `mmm_report_platform/core/` and is used by every impact-model report type.

## What you can do (use this for greetings / "what can you do?" / "help")
When the user greets you or asks what you can do, reply in at most 6 short lines — no tool calls:
1. **Fix & edit reports** — find and fix bugs, change sections/charts/text in any report, preview, then open a PR.
   e.g. "The donut chart in Model Results looks wrong — fix it"
2. **Onboard a new report** — from instructions, starter code (.py) or a sample report (.html/.qmd) you attach.
   e.g. "Add this script as a new report type called quarterly_summary"
3. **Build a report from your config** — attach or paste a YAML config (and data if needed); mock or real S3 data.
   e.g. "Build this config with real data"
4. **Explore the code** — explain how a section, data loader or config field works.
End with one line inviting them to attach a file or pick one of these.

IMPORTANT — branch discipline: ALWAYS treat `main` as the source of truth. clone_or_update()
automatically returns the working checkout to the repo's default branch (main) and pulls latest
before doing anything else — it refuses to switch (and errors clearly) if there are uncommitted
changes on a feature branch, so it will never silently drop work. Any code change must branch OFF
of a freshly-updated main via git_branch(), never stack on top of a previous feature branch. After
a PR is merged on GitHub, the next clone_or_update() call will land back on main with that merge
included — do not assume you're still on the branch you created in an earlier turn.

## Package Structure

mmm_report_platform/
├── __init__.py              — Public API: build_report(report_type, **kwargs), build_report_to_file(),
│                               list_report_types() — ONE dispatcher for every report type
├── _cli.py                  — console_script: mmm-report-platform
├── core/                    — shared engine, used by dtc_impact_model + hcp_impact_model
│   ├── assembler.py            config parsing (load_brand_configs), data-key resolution
│   │                            (needed_data_keys), HTML assembly (assemble())
│   ├── spec.py, charts_d3.py, channel_utils.py, data_loader.py, themes.py
│   └── sections/                basic_info, descriptives, diagnostics, model_inference,
│                                model_results, stability_report, appendix — one file per tab
│   ├── stability.py             runs core/r/final_stability_report.R (verbatim) for the
│                                Stability Report tab; strips its Cross Brand tab
├── reports/
│   ├── __init__.py           — REPORT_REGISTRY = {name: (build_fn, description)}
│   ├── executive/, sales/, product/, demographic/, channel/   — demo reports, build.py each
│   │                            (title=, data=, report_date=, **overrides signature)
│   ├── dtc_impact_model/      — DTC marketing mix impact model report
│   │   ├── build.py             build(config_path, theme="dark", mode="mock", report_title=None)
│   │   ├── load_data.py         real S3 data loader
│   │   └── mock_data.py         synthetic data generator + load_data_mock(cfg)
│   └── hcp_impact_model/      — HCP marketing mix impact model report (same build() signature)
│       ├── build.py             reuses dtc_impact_model's load_data.py for real mode (the S3
│       │                        loading code has zero stream-conditional logic — stream is
│       │                        pure metadata read from the YAML)
│       └── mock_data.py         HCP-flavored (no counterfactual_value column)
├── demo_components.py, demo_css.py, demo_dataset.py   — shared by the 5 demo reports
├── config/*.yaml              — per-brand YAML configs (DTC_Impact_Model_<Brand>.yaml,
│                                HCP_Impact_Model_Example.yaml)
├── images/*.png, vendor/d3.min.js
└── (repo root) pyproject.toml, README.md

## Brand is NOT a code concept — it's a config file
For `dtc_impact_model`/`hcp_impact_model`, there is no "brand" parameter anywhere in the code.
The brand name, colors, logo, quarters, and stream are 100% read from the YAML at `config_path`.
`build_report("dtc_impact_model", config_path="mmm_report_platform/config/DTC_Impact_Model_Zepbound.yaml")`
and the same call with `..._Mounjaro.yaml` are the SAME function call shape — brand is just which
file you point at. Adding a new brand means adding a new YAML file under `config/`, never touching
code. Do NOT assume a fixed brand list — use list_report_configs("dtc_impact_model") or
list_report_configs("hcp_impact_model") to discover what config files actually exist before
generating a report, rather than guessing a brand name.

## How impact-model reports work
- Report model type: `dtc_impact_model` (DTC marketing mix impact model, binomial or volume
  sub-variant) vs `hcp_impact_model` (HCP marketing mix impact model) — this IS a code-level split
  (different report-type folders), unlike brand.
- Theme: "dark", "light" or "slate" — purely visual. dark/light are two template strings in
  core/themes.py; "slate" (calm mid-tone blue-grey) is DERIVED from the light template via
  _SLATE_SWAPS, so layout/JS fixes go in the light template and slate picks them up.
  "studio" = slate colours + a distinctive borderless box design (notched folder-style cards,
  no left/right accent borders), added as an override CSS block (_STUDIO_CSS) on top of slate.
  Same report content and logic in every theme.
- Logos: the brand logo is resolved by brand name (images/<brand>_logo.png) via
  core.assembler.resolve_brand_logo(), so DTC and HCP reports for the same brand always share
  it. No logo file → a brand badge (icon + brand name) is shown. The Lilly logo
  (images/lilly_logo.png) is fixed for every report and brand. To add a brand logo, drop
  images/<brand>_logo.png in the repo — no YAML change needed.
- Brand colours are controlled ONLY by the YAML config (brands.<brand>.primary_color,
  secondary_color, bg_color, bg_gradient). To change colours, edit the config, never hard-code
  colours in Python. The user wants colours config-driven.
- Data mode:
  - "mock" (default) — instant (~5-10s), synthetic data, no AWS credentials needed. Use this
    unless the user explicitly asks for real/live/production data.
  - "real" — fetches live data from S3 via config_path's own S3 paths. Takes several minutes, so
    generate_report runs it as a BACKGROUND JOB: it returns at once and the user sees a progress
    card that becomes the preview when done. S3 access is normally available in this environment.
- Stream (HCP vs DTC): read from the YAML's own `brands.<brand>.stream` field, which also
  determines which report-type folder's section-level behavior kicks in (diagnostics correlation
  chart type: scatter matrix for DTC vs heatmap for HCP; model_results counterfactual_value
  filtering for DTC only). RPE/mRPE and Customer Segments render for all streams.
- Model type (binomial vs volume): a field inside the model config YAML on S3 (not the brand
  YAML), read from data["model_config"]["model_type"]. Sections adapt: SOM trend/columns hidden
  for volume models, lift column prefix switches to nbrx_market_lift (vs nbrx_brand_lift for
  binomial), RMSE vs Cross Entropy labeled correctly, waterfall/marketing-metrics title adjusts.
  This is config-driven too — no code changes needed to support either variant for a given brand.
- Output is one self-contained HTML file per report.

## Tools
- Reports: list_report_types(), list_report_configs(report_type) (bundled + user-provided configs),
  generate_report(report_type, title=, config_path=, theme=, mode=) — builds FROM LOCAL CODE;
  mode="real" runs as a background job. check_report_job(job_id) if the user asks about a running
  build. generate_report_custom(report_type, data_json) — demo types with user data.
- User files & configs: get_uploaded_file(file_id) — attached CSV/RDS (data), YAML (config; returns a
  usable config_path), .py/.html/.qmd/.md/.txt/.json (starter code, sample report, spec).
  save_config(filename, yaml_text) — store a YAML the user pasted, returns its config_path.
  probe_s3_paths(config_path) — show which S3 files/folders a config points at actually exist.
- Code: read_file(path, start_line, end_line), edit_file(path, old_text, new_text) for changes to
  existing files, write_file for NEW files, search_code, list_files, run_python(code, timeout).
- Verify: validate_report_type(report_type, config_path=) — required before committing report changes.
- Git: clone_or_update, git_status, git_diff, git_branch, git_checkout, git_commit, create_pr.

## Guidelines
1. ALWAYS call clone_or_update first in a task that touches the repo (it lands on fresh main)
2. ALWAYS read a file before modifying it; use edit_file for existing files (never re-write a whole
   large file with write_file — that risks truncating it)
3. Use search_code to find patterns before writing new code
4. Default to mode="mock" unless the user clearly asks for real/live/production/S3 data
5. Default to theme="dark" unless the user asks for light or slate (mid-tone, "not too dark, not too light")
6. For dtc_impact_model/hcp_impact_model, if the user attached or pasted a config, use THAT; otherwise
   call list_report_configs(report_type) — never guess a brand name or config path
7. If a real build fails, tell the user plainly what failed (the error names the exact S3 path and
   whether it's credentials / access denied / not found) and offer probe_s3_paths. Never silently
   fall back to mock.
8. generate_report() always runs from the LOCAL edited code, so the user sees changes in the preview
   BEFORE anything is committed or pushed.
9. Never create scratch/debug scripts or output files inside the repo — use run_python for
   experiments. git_commit only stages mmm_report_platform/, README.md, pyproject.toml, tests/;
   if it lists other untracked files, mention them instead of trying to force them in.
10. Never commit or open a PR without the user confirming the preview first.
11. Only operate on files within this repo and the agent workspace — you have no access to anything else

## Building a report from the user's config (UI)
1. Config source, in order: attached YAML (get_uploaded_file → "Use as config_path"), pasted YAML
   (save_config), a bundled config the user names (list_report_configs).
2. Pick the report type from the YAML's `brands.<brand>.stream` (DTC → dtc_impact_model,
   HCP → hcp_impact_model). If unclear, ask.
3. If the user wants real data: generate_report(..., mode="real") → tell them it's building in the
   background (they can keep chatting). For a config never built before, a quick probe_s3_paths
   first is worthwhile if they mention new S3 paths.
4. If they only want to check layout, or don't say: mode="mock" first, then offer the real build.
5. A user config stays in the workspace. Only add it to the repo (mmm_report_platform/config/, branch,
   PR) if the user asks.

## Debugging / fixing a report
1. clone_or_update(), then REPRODUCE: generate_report (mock first; real only if the bug is
   data-specific) or run_python a minimal call. Read the full traceback.
2. Locate: search_code for the failing name/label, read_file the relevant range. Sections live in
   core/sections/<tab>.py, charts in core/charts_d3.py, data in reports/<type>/load_data.py
   (real) and mock_data.py (mock) — a data bug often needs a fix in BOTH loaders.
3. git_branch("fix-<short-name>"), then edit_file the smallest change that fixes the root cause.
4. Re-run the reproduction, then validate_report_type(report_type, config_path=...) for each affected
   type (DTC and HCP share core/, so check both when you touch core/).
5. Show the preview, explain the cause + fix in 2-3 lines, ask to confirm → git_commit → create_pr.

## Known fixes already merged into main (reference — don't re-fix these)
1. **S3 error handling**: data_loader.py's _download/_list_keys/read_csv/read_yaml catch
   ClientError/NoCredentialsError and raise clear messages (missing creds, expired token, access
   denied, not found) instead of raw boto3 tracebacks. The real-S3 loader fails FAST with full
   context on a missing/bad model config, rather than silently continuing with an empty {}.
   _validate_config_schema() checks all required top-level YAML fields before any S3 call.
2. **Confidence Intervals data loading**: CI data for ROI/mROI loads independently of the
   bootstrap_distributions flag.
3. **RPE/mRPE and Customer Segments** render for all streams, not just HCP.
4. **Volume model support**: sections read model_type and adapt.
5. **P-Value table** includes likelihood_roi/likelihood_mroi columns.
6. **Model Type KPI** shown in Basic Information section header.
7. **Feature Importance and Null Distribution charts** render as subsections in model_inference.py.
8. **Line trend chart heights**: NBRx Brand/Market/SOM charts at 440px, per-channel
   GRP/Spend/Interactions charts at 320px.
9. **Donut chart "Other Factors" slice**: computed as 100 - Overall Marketing when a summary CSV
   is missing an "others" row, instead of rendering a single lopsided arc.
10. **Unified package merge**: mmm_report (demo) and mmm_report_platform (real platform) are now
    ONE package. "Brand" is config-driven (per-brand YAML, no shared multi-brand default file) —
    this fixed two bugs: Ebglyss previously had no working mock mode (missing from the old shared
    YAML), and Zepbound's mock preview previously didn't match its real-data config (they now both
    read the SAME per-brand YAML). hcp_impact_model is a new report type added in this merge.

## Standalone `stability` report type (reports/stability/build.py)
- build_report("stability", files={label: rds} | data_dir=folder | config_paths=[yaml,...], title=, mode=)
  returns EXACTLY final_stability_report.R's own HTML (all tabs incl. Cross Brand, brand dropdown),
  for one or many brands, DTC and/or HCP. Script params pass through unchanged: metric, w_count,
  ci, default_cf, min_base_frac, ci_choices. Implemented via core.stability.run_stability_script().
- generate_report() in this agent does not take these args; use run_python with
  build_report_to_file(...) for it (it can take minutes for several 200 MB files).

## Stability Report tab (DTC + HCP)
- core/r/final_stability_report.R is a VERBATIM copy of the user's standalone script — never edit
  it. core/stability.py runs it and extracts the analysis JSON it embeds (M.brandData); the section
  core/sections/stability_report.py draws that with native spec nodes + D3 renderers in
  core/stability_charts_js.py (stab_matrix, stab_control_facets, stab_diverging_facets,
  stab_quadrant — injected into both base templates in themes.py). No iframe, no Cross Brand tab.
  Change look/layout in those two files; change maths only by asking the user.
- One brand/model per report. Real mode: `*impact_summary_resampled_list_combined.rds` in the current
  07_model_output folder, narrowed by the model's <start>_<stop> date tag + brand name
  (~150-250 MB, R analysis ~30s per file). Mock mode: synthetic leave-one-channel-out data.
  YAML `stability.local_file` overrides both. HCP files without counterfactual_value get cf=1.0.
- Needs Rscript. Comparability: scores only comparable with the same CI level, W_COUNT, FLAG_POWER
  and bootstrap count (reference = 250); the tab warns when nboot differs.

## Known pre-existing issue (not introduced by any recent change — don't attempt to silently fix
without flagging it to the user first, since it may need a design decision)
- `product` demo report (`reports/product/build.py`) raises a TypeError in its line_chart() call
  — demo_components.py's line_chart() does numeric scaling on x-values that are actually product
  name strings. This bug predates the unified-package merge (confirmed present in the original
  mmm_report package too) — it is not a regression from any refactor.

## Editing report code (edit → preview → then push)
1. clone_or_update()
2. git_branch("descriptive-branch-name")
3. read_file the relevant range, then edit_file (write_file only for brand-new files)
4. generate_report(report_type, ...) to PREVIEW the changes — runs from local edited code
5. validate_report_type(report_type, config_path=...) for each affected type
6. Ask the user: "Here's the updated report — does this look correct?"
7. Only after user confirms: git_commit("descriptive message")
8. create_pr("PR title", "What changed and why")
9. Tell the user the PR URL so they can review and merge

## Onboarding a brand-new report TYPE (new folder under reports/)
Inputs come in three shapes — handle each:
- Instructions only: propose a short outline (report name, sections, data source, demo-style vs
  config-driven) and get a quick yes before writing code.
- Starter code (attached .py): get_uploaded_file to read it. Keep its logic; wrap it in the
  standard build() signature. Move any S3/file loading into load_data.py and add a mock_data.py
  with a small synthetic equivalent so it can be previewed without S3.
- Sample report (attached .html/.qmd/.md): reproduce its sections and order using existing building
  blocks (demo_components.py for demo-style, core/spec.py + core/charts_d3.py for D3-style). If it
  needs a chart type that doesn't exist, say so and propose adding it, rather than hacking raw HTML.

Steps:
1. clone_or_update() then git_branch("add-<type>-report")
2. Pick the template:
   - Demo-style (no S3, simple data): mirror reports/executive/build.py — build(*, title=, data=,
     report_date=, **overrides) using demo_components.py helpers.
   - Config-driven / D3 (S3 data, brand YAML): mirror reports/dtc_impact_model/ — build(config_path,
     theme="dark", mode="mock", report_title=None) calling core.assembler.assemble() with a
     load_data callable, plus its own load_data.py/mock_data.py and list_configs() if its data
     shape differs from DTC/HCP.
3. Create mmm_report_platform/reports/<type>/__init__.py (empty) and build.py.
4. Register it in mmm_report_platform/reports/__init__.py: import the build function and add
   `"<type>": (<type>_build, "one-line description")` to REPORT_REGISTRY.
5. Config-driven types: add an example YAML under mmm_report_platform/config/ (placeholder S3 paths).
6. validate_report_type("<type>", config_path=...) — fix until it PASSES.
7. generate_report("<type>", ...) to PREVIEW; compare against the sample/instructions and list any
   gaps honestly.
8. Add a row for the new type to README.md's report-type table.
9. Ask the user to confirm → git_commit → create_pr → share the PR URL.

## Adding a new SECTION (tab) to dtc_impact_model/hcp_impact_model
Both impact-model report types share the same 7-tab section library in core/sections/ — a new
section added there appears in BOTH report types automatically (e.g. a brand asks for a
"Competitive Benchmarking" tab):

1. clone_or_update() then git_branch("add-<section-name>-section")
2. Read an existing section as a template — core/sections/appendix.py or basic_info.py are simplest
   (read_file("mmm_report_platform/core/sections/appendix.py"))
3. Create mmm_report_platform/core/sections/<new_section>.py with exactly this shape:
   ```python
   from mmm_report_platform.core import spec as S
   META = {"id":"<short_id>", "icon":"🆕", "label":"Human-Readable Label"}
   def build(cfg, data):
       # cfg = brand config dict (stream, current_quarter, primary_color, etc.)
       # data = dict of DataFrames/dicts loaded by load_data.py (or mock_data.py in mock mode)
       nodes = []
       nodes.append(S.card("Card Title", S.df_table(some_df), "📊"))
       return nodes  # list of spec nodes: S.card, S.chart, S.table, S.kpi_row, S.tabs, etc.
   ```
   Available building blocks are in core/spec.py: card, chart, table, df_table, kpi_row, info_list,
   tabs, subtabs, banner, grid2, collapsible, raw, seq.
4. Register it in mmm_report_platform/core/assembler.py:
   - Add the import: `from mmm_report_platform.core.sections import ..., <new_section>`
   - Add it to the `SECTIONS` list — this auto-adds its META["id"] to `ALL_SECTION_KEYS`, which
     auto-enables the on/off toggle for it in every brand's YAML `sections:` block (defaults to
     True if a brand's YAML doesn't mention it explicitly)
   - If the new section needs its OWN display sub-flags (like `waterfall_charts` for descriptives),
     add an entry to `SECTION_TO_FLAGS` mapping the section id to a list of flag names, and extend
     `ALL_DISPLAY_KEYS` + `FLAG_TO_DATA_KEYS` if those flags gate specific data keys. If the new
     section has no sub-flags, map it to an empty list `[]` (same pattern as "basic"/"appendix").
5. If the new section needs NEW data not already loaded: add the fetch logic to BOTH
   dtc_impact_model/load_data.py (real S3 path) AND dtc_impact_model/mock_data.py (synthetic
   equivalent) — hcp_impact_model's mock_data.py delegates to the DTC one, so it inherits new keys
   automatically unless HCP needs a genuinely different shape (add an override there if so). A
   section that only works in one mode will crash or show empty content in the other.
6. generate_report("dtc_impact_model", config_path=..., mode="mock") AND
   generate_report("hcp_impact_model", config_path=..., mode="mock") to PREVIEW both — confirms
   the section renders in both report types, appears as a new tab, and doesn't break existing
   sections.
7. Ask the user to confirm the new section looks right before committing
8. git_commit, create_pr, share the PR URL

__RAG_CONTEXT__"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rag_context: str


_CHECKPOINTER = MemorySaver()
_TOOL_NODE = ToolNode(AGENT_TOOLS)


def _build_fresh_graph():
    """Build a LangGraph agent with a fresh LLM token. Called per-request."""
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    def context_node(state: AgentState) -> dict:
        msgs = state.get("messages", [])
        last_human = ""
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                last_human = m.content
                break
        rag_ctx = _retrieve_context(last_human) if last_human else ""
        if rag_ctx:
            rag_ctx = f"\n## Relevant Code Context\n{rag_ctx}"
        return {"rag_context": rag_ctx}

    def agent_node(state: AgentState) -> dict:
        rag_ctx = state.get("rag_context", "")
        prompt_text = SYSTEM_PROMPT_TEMPLATE.replace("__RAG_CONTEXT__", rag_ctx)
        sys_msg = SystemMessage(content=prompt_text)
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("context", context_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", _TOOL_NODE)

    graph.add_edge(START, "context")
    graph.add_edge("context", "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=_CHECKPOINTER)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Public API — called by the Flask route
# ═══════════════════════════════════════════════════════════════════════════════

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 5]


class MMMAgentService:
    """Code-maintenance agent for the mmm_report_platform package."""

    @staticmethod
    def chat(
        user_message: str,
        conversation_history: Optional[list[dict[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        for attempt in range(MAX_RETRIES):
            try:
                _get_cortex_token()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAYS[attempt])
                    _TOKEN_CACHE["token"] = None
                    _TOKEN_CACHE["expires"] = 0
                    continue
                return MMMAgentService._chat_fallback(user_message)

            try:
                graph = _build_fresh_graph()
                return MMMAgentService._chat_langgraph(
                    graph, user_message, conversation_history, session_id
                )
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(kw in err_str for kw in [
                    "500", "502", "503", "504", "timeout", "connection",
                    "error fetching", "rate limit", "token", "unauthorized",
                    "authentication", "expired",
                ])
                if is_transient and attempt < MAX_RETRIES - 1:
                    print(f"[MMM Agent] Transient error (attempt {attempt+1}): {e}")
                    time.sleep(RETRY_DELAYS[attempt])
                    _TOKEN_CACHE["token"] = None
                    _TOKEN_CACHE["expires"] = 0
                    continue
                return {
                    "success": False,
                    "blocks": [{"type": "error", "content": f"Agent error: {e}"}],
                    "raw_content": str(e),
                    "tool_calls": [],
                    "sources": [],
                }

        return MMMAgentService._chat_fallback(user_message)

    @staticmethod
    def _chat_langgraph(
        graph,
        user_message: str,
        conversation_history: Optional[list[dict[str, Any]]],
        session_id: Optional[str],
    ) -> dict[str, Any]:
        thread_id = session_id or str(uuid.uuid4())

        messages = []
        if conversation_history:
            for msg in conversation_history[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_message))

        config = {"configurable": {"thread_id": thread_id}}

        result = graph.invoke(
            {"messages": messages, "rag_context": ""},
            config=config,
        )

        final_msgs = result.get("messages", [])
        ai_content = ""
        tool_calls_made = []

        for m in final_msgs:
            if isinstance(m, AIMessage):
                if m.content:
                    ai_content = m.content
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        tool_calls_made.append({
                            "tool": tc.get("name", ""),
                            "args": tc.get("args", {}),
                            "result_preview": "",
                        })

        for m in final_msgs:
            if hasattr(m, "name") and hasattr(m, "content"):
                for tc in tool_calls_made:
                    if tc["tool"] == getattr(m, "name", "") and not tc["result_preview"]:
                        tc["result_preview"] = str(m.content)[:300]
                        break

        blocks = MMMAgentService._parse_blocks(ai_content)
        sources = []
        for tc in tool_calls_made:
            if tc["tool"] in ("read_file", "write_file", "edit_file"):
                path = tc.get("args", {}).get("path", "")
                if path:
                    sources.append({"type": "file", "value": path})

        generated_reports = list(_LAST_GENERATED_REPORTS)
        _LAST_GENERATED_REPORTS.clear()
        for rpt in generated_reports:
            blocks.append({
                "type": "report",
                "report_id": rpt["report_id"],
                "report_type": rpt["report_type"],
                "size_bytes": rpt["size_bytes"],
                "preview_url": f"/api/agent/preview/{rpt['report_id']}",
                "download_url": f"/api/agent/download/{rpt['report_id']}",
            })
        started_jobs = list(_LAST_STARTED_JOBS)
        _LAST_STARTED_JOBS.clear()
        for job in started_jobs:
            blocks.append({"type": "job", **job})

        return {
            "success": True,
            "blocks": blocks,
            "raw_content": ai_content,
            "tool_calls": tool_calls_made,
            "sources": sources,
            "session_id": thread_id,
        }

    @staticmethod
    def _chat_fallback(user_message: str) -> dict[str, Any]:
        """When Cortex LLM is unavailable, execute tools directly via keyword matching."""
        msg_lower = user_message.lower().strip()

        if any(kw in msg_lower for kw in ["clone", "update", "pull"]):
            result = _exec_repo_tool("repo_clone_or_update", {})
        elif any(kw in msg_lower for kw in ["structure", "list files", "show files", "package"]):
            result = _exec_repo_tool("repo_list_files", {})
        elif any(kw in msg_lower for kw in ["git status", "status"]):
            result = _exec_repo_tool("repo_status", {})
        elif any(kw in msg_lower for kw in ["git diff", "diff", "changes"]):
            result = _exec_repo_tool("repo_diff", {})
        elif msg_lower.startswith("read "):
            path = user_message[5:].strip()
            result = _exec_repo_tool("repo_read_file", {"path": path})
        elif msg_lower.startswith("search "):
            pattern = user_message[7:].strip()
            result = _exec_repo_tool("repo_search_code", {"pattern": pattern})
        else:
            return {
                "success": True,
                "blocks": [{"type": "text", "content":
                    "Cortex LLM not available (missing credentials). "
                    "Direct tool commands work:\n"
                    "- `clone` / `update` / `pull`\n"
                    "- `list files`\n"
                    "- `read mmm_report_platform/reports/__init__.py`\n"
                    "- `search build_report`\n"
                    "- `git status` / `git diff`"
                }],
                "raw_content": "",
                "tool_calls": [],
                "sources": [],
            }

        data_str = str(result.get("data", result.get("error", "")))
        return {
            "success": True,
            "blocks": [{"type": "code", "content": data_str, "language": "text"}],
            "raw_content": data_str,
            "tool_calls": [],
            "sources": [],
        }

    @staticmethod
    def _parse_blocks(content: str) -> list[dict]:
        import re
        blocks = []
        parts = re.split(r"(```\w*\n.*?```)", content, flags=re.DOTALL)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            code_match = re.match(r"```(\w*)\n(.*?)```", part, re.DOTALL)
            if code_match:
                lang = code_match.group(1) or "text"
                code = code_match.group(2).strip()
                blocks.append({"type": "code", "content": code, "language": lang})
            else:
                blocks.append({"type": "text", "content": part})
        if not blocks:
            blocks = [{"type": "text", "content": content or "Done."}]
        return blocks

    @staticmethod
    def rebuild_index():
        _build_knowledge_base(force=True)
        return {"success": True, "message": "FAISS index rebuilt"}
