"""mmm_agent_service.py — Code-maintenance agent for the unified mmm_report_platform package.

Uses LangChain + LangGraph with Cortex (OpenAI-compatible gateway):
  - LangGraph tool-calling agent with file/git tools for Kuldeepsvts/report.git,
    which contains the mmm_report_platform package (demo reports + DTC/HCP
    impact model reports, each report type in its own reports/<type>/ folder)
  - FAISS RAG over the mmm_report_platform source code for context
  - Multi-turn memory via MemorySaver checkpointer
  - Token refresh per-request to handle SSO expiry
  - Automatic retry on transient LLM failures
"""

import hashlib
import json as _json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Annotated, Literal, Optional

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
from services.repo_tools import (
    execute_tool as _exec_repo_tool,
    EXTERNAL_REPO_DIR,
)

_PREVIEW_CACHE: dict[str, str] = {}
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
# 2. LangChain Tools — all scoped to github.com/Kuldeepsvts/report.git
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
def read_file(path: str) -> str:
    """Read a file from the mmm_report_platform repo. Returns content with line numbers."""
    r = _exec_repo_tool("repo_read_file", {"path": path})
    return r.get("data", r.get("error", "unknown error"))


@tool
def write_file(path: str, content: str) -> str:
    """Write or overwrite a file in the mmm_report_platform repo. Creates parent dirs if needed."""
    r = _exec_repo_tool("repo_write_file", {"path": path, "content": content})
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
    """Stage all changes and commit with the given message."""
    r = _exec_repo_tool("repo_commit", {"message": message})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_push() -> str:
    """Push the current branch to origin."""
    r = _exec_repo_tool("repo_push", {})
    return r.get("data", r.get("error", "unknown error"))


def _run_report_subprocess(code: str, timeout: int = 30) -> tuple[str, str]:
    """Run code in a subprocess against the local repo. Returns (html, error).
    Auto-clones the repo if it doesn't exist yet."""
    if not (EXTERNAL_REPO_DIR / ".git").exists():
        clone_result = _exec_repo_tool("repo_clone_or_update", {})
        if not clone_result.get("success"):
            return "", f"Failed to clone repo: {clone_result.get('error')}"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(EXTERNAL_REPO_DIR),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", f"Timed out after {timeout}s"
    if result.returncode != 0:
        return "", result.stderr.strip()
    html = result.stdout
    if not html.strip():
        return "", f"Empty output. stderr: {result.stderr.strip()}"
    return html, ""


def _cache_report(html: str, report_type: str) -> str:
    """Cache generated HTML and return the report_id."""
    report_id = hashlib.sha256((html + str(time.time())).encode()).hexdigest()[:12]
    _PREVIEW_CACHE[report_id] = html
    _LAST_GENERATED_REPORTS.append({
        "report_id": report_id,
        "report_type": report_type,
        "size_bytes": len(html),
        "html": html,
    })
    return report_id


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
    """List available YAML config files for an impact-model report type.

    Use this BEFORE generate_report() for dtc_impact_model/hcp_impact_model —
    these report types require a config_path, and this tells you which brand
    configs actually exist (e.g. config/DTC_Impact_Model_Zepbound.yaml) instead
    of guessing a brand name. There is no fixed brand list — any YAML dropped
    into mmm_report_platform/config/ following the existing schema works.
    """
    code = (
        "import sys; sys.path.insert(0, '.'); "
        f"from mmm_report_platform.reports.{report_type}.build import list_configs; "
        "r = list_configs(); "
        "[print(f'  {n:45s} — {d}') for n, d in r.items()]; "
        "print(f'{len(r)} config(s) available')"
    )
    out, err = _run_report_subprocess(code)
    if err:
        return f"Error: {err}"
    return out


@tool
def generate_report(
    report_type: str = "executive",
    title: str = "",
    config_path: str = "",
    theme: Literal["dark", "light"] = "dark",
    mode: Literal["mock", "real"] = "mock",
) -> str:
    """Generate an HTML report from the LOCAL repo code and return a preview.

    This runs the report from the locally edited code in
    external_repos/report/mmm_report_platform/, NOT from any pip-installed
    version — so if you just edited a report module, calling this shows the
    updated version immediately.

    Use list_report_types() first to see all options.

    For demo report types (executive, sales, product, demographic, channel):
        generate_report("executive")
        generate_report("sales", title="Q3 Sales")

    For impact-model report types (dtc_impact_model, hcp_impact_model):
        config_path is REQUIRED — brand, colors, quarters, and stream are all
        read from that YAML file, not passed as separate arguments. Use
        list_report_configs(report_type) first to find available config
        files; there is no fixed brand list to guess from.
            generate_report("dtc_impact_model", config_path="mmm_report_platform/config/DTC_Impact_Model_Zepbound.yaml")
            generate_report("dtc_impact_model", config_path="...", theme="light", mode="real")

    Args:
        report_type: One of list_report_types()'s keys.
        title: Demo reports only — overrides the default title.
        config_path: Impact-model reports only — REQUIRED. Path to a brand
            YAML config, relative to the repo root (e.g.
            "mmm_report_platform/config/DTC_Impact_Model_Zepbound.yaml").
        theme: Impact-model reports only — "dark" or "light" (visual only).
        mode: Impact-model reports only — "mock" (fast, synthetic, default)
            or "real" (fetches live S3 data, needs AWS credentials, ~30-60s).
    The UI will automatically show preview and download buttons.
    """
    is_impact_model = report_type in ("dtc_impact_model", "hcp_impact_model")

    if is_impact_model and not config_path:
        return (
            f"Error: {report_type} requires config_path. "
            f"Call list_report_configs('{report_type}') to see available configs."
        )

    code = (
        f"import sys; sys.path.insert(0, '.'); "
        f"from mmm_report_platform import build_report; "
        f"html = build_report({report_type!r}"
    )
    if is_impact_model:
        pkg_mode = "s3" if mode == "real" else "mock"
        code += f", config_path={config_path!r}, theme={theme!r}, mode={pkg_mode!r}"
    elif title:
        code += f", title={title!r}"
    code += "); sys.stdout.write(html)"

    html, err = _run_report_subprocess(code, timeout=90 if is_impact_model else 30)
    if err:
        return f"Error generating report: {err}"

    report_id = _cache_report(html, report_type)
    return (
        f"Successfully generated {report_type} report ({len(html):,} bytes) from local code. "
        f"The preview and download buttons are now visible to the user."
    )


@tool
def create_pr(title: str, body: str = "") -> str:
    """Create a GitHub Pull Request from the current branch to main.

    IMPORTANT: You must be on a feature branch (not main). The workflow is:
    1. git_branch("my-feature")
    2. write_file / edit code
    3. git_commit("description of changes")
    4. create_pr("PR title", "Description of what changed and why")

    This pushes the branch and opens the PR on GitHub automatically.
    """
    r = _exec_repo_tool("repo_create_pr", {"title": title, "body": body})
    return r.get("data", r.get("error", "unknown error"))


@tool
def run_script(script: str) -> str:
    """Run a Python script file inside the mmm_report_platform repo.

    Use this to verify changes work before committing.
    Returns stdout, stderr, and exit status. Timeout: 30 seconds.
    """
    r = _exec_repo_tool("repo_run_script", {"script": script})
    return r.get("data", r.get("error", "unknown error"))


@tool
def run_python(code: str) -> str:
    """Run an inline Python code snippet inside the mmm_report_platform repo.

    Use this to quickly verify imports, test functions, or check output.
    The repo root is on PYTHONPATH so you can import mmm_report_platform directly.

    Returns stdout, stderr, and exit status. Timeout: 30 seconds.
    """
    r = _exec_repo_tool("repo_run_python", {"code": code})
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
    """Retrieve data from a file the user uploaded (CSV or RDS).

    When a user attaches a file, the chat message will contain a file_id.
    Call this tool with that file_id to get the parsed data as JSON.
    Returns the filename, columns, row count, and data rows (up to 500).
    Use this data with generate_report_custom to build a report from the user's file.
    """
    entry = _UPLOADED_FILES.get(file_id)
    if not entry:
        return f"Error: file_id '{file_id}' not found. It may have expired."
    data_str = _json.dumps(entry["data"][:20], default=str)
    return (
        f"File: {entry['filename']}\n"
        f"Columns: {', '.join(entry['columns'])}\n"
        f"Rows: {entry['rows']}\n"
        f"First 20 rows (JSON):\n{data_str}"
    )


AGENT_TOOLS = [
    clone_or_update, list_files, read_file, write_file, search_code,
    git_status, git_diff, git_branch, git_checkout, git_commit, git_push,
    generate_report, generate_report_custom, list_report_types, list_report_configs,
    create_pr, run_script, run_python, get_uploaded_file,
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

SYSTEM_PROMPT_TEMPLATE = """You are a report-platform maintenance agent. You maintain ONE git repo
(https://github.com/Kuldeepsvts/report.git) containing ONE unified Python package: `mmm_report_platform`.
Every report TYPE lives in its own folder under `mmm_report_platform/reports/<type>/`, each exposing
a `build(...)` function. A shared rendering engine (config parsing, D3/HTML assembly, the 6-tab
section library) lives in `mmm_report_platform/core/` and is used by every impact-model report type.

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
│                                model_results, appendix — one file per report tab
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
- Theme: "dark" or "light" — purely a visual/CSS difference (two template strings in
  core/themes.py). Same report content and logic either way.
- Data mode:
  - "mock" (default) — instant (~5-10s), synthetic data, no AWS credentials needed. Use this
    unless the user explicitly asks for real/live/production data.
  - "real" — fetches live data from S3 (~30-60s) via config_path's own S3 paths. Requires valid
    AWS credentials configured in this environment (env vars, ~/.aws/credentials, or SSO).
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
- list_report_types() — see all 7 registered report types + descriptions
- list_report_configs(report_type) — for dtc_impact_model/hcp_impact_model, see what YAML config
  files actually exist in config/ (always call this before guessing a config_path)
- generate_report(report_type, title=, config_path=, theme=, mode=) — builds and previews FROM
  LOCAL CODE. For demo types, pass title= (optional). For impact-model types, config_path is
  REQUIRED; theme/mode are optional (default dark/mock).
- generate_report_custom(report_type, data_json, title=) — demo types only, builds from
  user-pasted custom data instead of the standard dataset.
- Same generic tools for inspecting/editing code: clone_or_update, read_file, write_file,
  search_code, git_branch, git_commit, create_pr, run_python.

## Guidelines
1. ALWAYS call clone_or_update first to ensure the repo is present, up-to-date, and on main
2. ALWAYS read a file before modifying it
3. Use search_code to find patterns before writing new code
4. Default to mode="mock" unless the user clearly asks for real/live/production/S3 data
5. Default to theme="dark" unless the user asks for light
6. For dtc_impact_model/hcp_impact_model, ALWAYS call list_report_configs(report_type) before
   generate_report if you don't already know the exact config_path — never guess a brand name
7. If mode="real" fails (credentials or S3 path issues), tell the user plainly what failed — don't
   silently fall back to mock without saying so. Error messages are detailed: they name the exact
   S3 path tried, distinguish credentials-missing vs expired vs access-denied vs not-found, and
   truncate long file listings to 20 entries + a count.
8. generate_report() always runs from the LOCAL edited code in
   external_repos/report/mmm_report_platform/ — so the user can see changes immediately in the
   preview BEFORE committing or pushing.
9. Only operate on files within this repo — you have no access to anything else

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

## Known pre-existing issue (not introduced by any recent change — don't attempt to silently fix
without flagging it to the user first, since it may need a design decision)
- `product` demo report (`reports/product/build.py`) raises a TypeError in its line_chart() call
  — demo_components.py's line_chart() does numeric scaling on x-values that are actually product
  name strings. This bug predates the unified-package merge (confirmed present in the original
  mmm_report package too) — it is not a regression from any refactor.

## Editing report code (edit → preview → then push)
1. clone_or_update()
2. git_branch("descriptive-branch-name")
3. Read the file, make edits with write_file
4. generate_report(report_type, ...) to PREVIEW the changes — runs from local edited code, NOT
   any installed/pip version
5. Ask the user: "Here's the updated report — does this look correct?"
6. Only after user confirms: git_commit("descriptive message")
7. create_pr("PR title", "What changed and why")
8. Tell the user the PR URL so they can review and merge

## Adding a brand-new report TYPE (new folder under reports/)
1. clone_or_update() then git_branch("add-<type>-report")
2. Create mmm_report_platform/reports/<new_type>/build.py:
   - Demo-style report (no S3, simple data): mirror reports/executive/build.py — a build(*, title=,
     data=, report_date=, **overrides) function using demo_components.py helpers.
   - Impact-model-style report (config-driven, D3-rendered): mirror reports/dtc_impact_model/ —
     a build(config_path, theme="dark", mode="mock", report_title=None) function that calls
     core.assembler.assemble() with a load_data callable, plus its own load_data.py/mock_data.py
     if it needs genuinely different data shapes from DTC/HCP.
3. Register in mmm_report_platform/reports/__init__.py: import the build function and add an entry
   to REPORT_REGISTRY: `"new_type": (new_type_build, "description")`
4. generate_report("new_type", ...) to PREVIEW — confirms it's discoverable via
   list_report_types() and renders correctly.
5. Ask the user to confirm before committing
6. git_commit, create_pr, share the PR URL

## Adding a new SECTION (tab) to dtc_impact_model/hcp_impact_model
Both impact-model report types share the same 6-tab section library in core/sections/ — a new
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
            if tc["tool"] in ("read_file", "write_file"):
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
