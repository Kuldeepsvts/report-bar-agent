"""mmm_agent_service.py — Code-maintenance agent for the mmm_report package.

Uses LangChain + LangGraph with Cortex (OpenAI-compatible gateway):
  - LangGraph tool-calling agent with file/git tools for Kuldeepsvts/report.git
  - FAISS RAG over the mmm_report source code for context
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
    """Clone the mmm_report repo if not present locally, else pull latest. Call this first before any other tool."""
    r = _exec_repo_tool("repo_clone_or_update", {})
    return r.get("data", r.get("error", "unknown error"))


@tool
def list_files(pattern: str = "") -> str:
    """List files in the mmm_report repo. Optional glob pattern e.g. '**/*.py'."""
    r = _exec_repo_tool("repo_list_files", {"pattern": pattern})
    return r.get("data", r.get("error", "unknown error"))


@tool
def read_file(path: str) -> str:
    """Read a file from the mmm_report repo. Returns content with line numbers."""
    r = _exec_repo_tool("repo_read_file", {"path": path})
    return r.get("data", r.get("error", "unknown error"))


@tool
def write_file(path: str, content: str) -> str:
    """Write or overwrite a file in the mmm_report repo. Creates parent dirs if needed."""
    r = _exec_repo_tool("repo_write_file", {"path": path, "content": content})
    return r.get("data", r.get("error", "unknown error"))


@tool
def search_code(pattern: str, file_pattern: str = "") -> str:
    """Search for a text/regex pattern across files in the mmm_report repo."""
    r = _exec_repo_tool("repo_search_code", {"pattern": pattern, "file_pattern": file_pattern})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_status() -> str:
    """Show git status of the mmm_report repo."""
    r = _exec_repo_tool("repo_status", {})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_diff(path: str = "") -> str:
    """Show git diff in the mmm_report repo. Optional: pass a file path."""
    r = _exec_repo_tool("repo_diff", {"path": path} if path else {})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_branch(name: str) -> str:
    """Create and switch to a new branch in the mmm_report repo."""
    r = _exec_repo_tool("repo_branch", {"name": name})
    return r.get("data", r.get("error", "unknown error"))


@tool
def git_checkout(name: str) -> str:
    """Switch to an existing branch in the mmm_report repo."""
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


def _run_report_subprocess(code: str) -> tuple[str, str]:
    """Run code in a subprocess against the local repo. Returns (html, error).
    Auto-clones the repo if it doesn't exist yet."""
    if not (EXTERNAL_REPO_DIR / ".git").exists():
        clone_result = _exec_repo_tool("repo_clone_or_update", {})
        if not clone_result.get("success"):
            return "", f"Failed to clone repo: {clone_result.get('error')}"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(EXTERNAL_REPO_DIR),
        capture_output=True, text=True, timeout=30,
    )
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
def generate_report(report_type: str = "executive", title: str = "") -> str:
    """Generate an HTML report from the LOCAL repo code and return a preview.

    This runs the report from the locally edited code in external_repos/report/,
    NOT from the pip-installed version. So if you just edited a report module,
    calling this will show the user the updated version immediately.

    Available report types: executive, sales, product, demographic, channel.
    Use list_report_types() first to see all options.
    The UI will automatically show preview and download buttons.
    """
    code = (
        f"import sys; sys.path.insert(0, '.'); "
        f"from mmm_report import build_report; "
        f"html = build_report({report_type!r}"
    )
    if title:
        code += f", title={title!r}"
    code += "); sys.stdout.write(html)"

    html, err = _run_report_subprocess(code)
    if err:
        return f"Error generating report: {err}"

    report_id = _cache_report(html, report_type)
    return (
        f"Successfully generated {report_type} report ({len(html):,} bytes) from local code. "
        f"The preview and download buttons are now visible to the user."
    )


@tool
def list_report_types() -> str:
    """List all available report types that can be generated."""
    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from mmm_report import list_reports; "
        "r = list_reports(); "
        "[print(f'  {n:15s} — {d}') for n, d in r.items()]; "
        "print(f'{len(r)} report types available')"
    )
    html, err = _run_report_subprocess(code)
    if err:
        return f"Error: {err}"
    return html


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
    """Run a Python script file inside the mmm_report repo.

    Use this to verify changes work before committing.
    Returns stdout, stderr, and exit status. Timeout: 30 seconds.
    """
    r = _exec_repo_tool("repo_run_script", {"script": script})
    return r.get("data", r.get("error", "unknown error"))


@tool
def run_python(code: str) -> str:
    """Run an inline Python code snippet inside the mmm_report repo.

    Use this to quickly verify imports, test functions, or check output.
    The repo root is on PYTHONPATH so you can import mmm_report directly.

    Returns stdout, stderr, and exit status. Timeout: 30 seconds.
    """
    r = _exec_repo_tool("repo_run_python", {"code": code})
    return r.get("data", r.get("error", "unknown error"))


@tool
def generate_report_custom(report_type: str, data_json: str, title: str = "") -> str:
    """Generate a report using custom user-provided data from the LOCAL repo code.

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
        f"from mmm_report import build_report; "
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
    generate_report, generate_report_custom, list_report_types,
    create_pr, run_script, run_python, get_uploaded_file,
]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAISS RAG knowledge base — index the mmm_report source code
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

SYSTEM_PROMPT_TEMPLATE = """You are the mmm_report package maintenance agent.
You maintain the `mmm_report` Python package at https://github.com/Kuldeepsvts/report.git —
a pip-installable multi-report HTML generator (zero external dependencies, Python 3.9+).

## Package Structure

mmm_report/
├── __init__.py       — Public API: report functions, build_report(name), list_reports(), DATASET
├── dataset.py        — Standard dataset (KPIS, REGIONS, COUNTRIES, PRODUCTS, ETHNICITIES, AGE_GROUPS, GENDERS, CHANNELS)
├── components.py     — Reusable HTML builders: kpi_cards, horizontal_bars, data_table, pie_chart, stat_card, stat_grid, section, page
├── css.py            — Shared CSS with base_css(accent) and color palettes
├── _cli.py           — CLI: mmm-report [type] [output.html]
└── reports/
    ├── __init__.py   — REPORT_REGISTRY dict, list_reports(), build(name)
    ├── executive.py  — executive_report(): KPIs + summary + revenue by region
    ├── sales.py      — sales_report(): KPIs, region bars, country table, product revenue
    ├── product.py    — product_report(): product ranking, margin analysis, top/bottom
    ├── demographic.py— demographic_report(): ethnicity pie, age bars, gender table
    └── channel.py    — channel_report(): channel spend, revenue, ROI analysis

Also at repo root: pyproject.toml, README.md, report.html

## Key Patterns
- Each report function: def xxx_report(*, title=..., data=None, report_date=None, **overrides) -> str
- data defaults to DATASET; overrides replace individual keys (e.g. kpis=[...])
- Reports compose components: section(), kpi_cards(), horizontal_bars(), data_table(), pie_chart(), stat_grid()
- page() wraps sections into a complete HTML doc with embedded CSS
- REPORT_REGISTRY in reports/__init__.py maps name -> (function, description)
- build_report(name) and build_report_to_file(path, name) use the registry

## Adding a New Report Type
1. Create mmm_report/reports/new_name.py with def new_name_report(*, title=..., data=None, report_date=None, **overrides) -> str
2. Use components from components.py (kpi_cards, data_table, horizontal_bars, pie_chart, stat_card, section, page)
3. Register in reports/__init__.py: import it and add to REPORT_REGISTRY
4. Export in __init__.py
5. Add new data keys to dataset.py if needed
6. Update README.md with the new report type, usage example, and data format

## Guidelines
1. ALWAYS call clone_or_update first to ensure the repo is present and up-to-date
2. ALWAYS read a file before modifying it
3. Use search_code to find patterns before writing new code
4. Follow existing code style
5. After changes, use run_python to verify the code works before committing
6. If asked to push, confirm changes look correct first (use git_diff)
7. Only operate on files within this repo — you have no access to anything else

## Workflows

### Generating a report
1. Use generate_report(report_type) — the UI shows preview/download buttons automatically
2. Use list_report_types() to see all available types
3. For custom data from the user, use generate_report_custom(report_type, data_json)

### Making code changes (edit → preview → then push)
1. clone_or_update()
2. git_branch("descriptive-branch-name")
3. Read the file, make edits with write_file
4. generate_report(report_type) to PREVIEW the changes — this runs from local edited code, NOT the pip-installed version
5. Ask the user: "Here's the updated report — does this look correct?"
6. Only after user confirms: git_commit("descriptive message")
7. create_pr("PR title", "What changed and why")
8. Tell the user the PR URL so they can review and merge

IMPORTANT: generate_report() always runs from the LOCAL edited code in external_repos/report/.
This means the user can see changes immediately in the preview BEFORE committing or pushing.
The workflow is: edit code → preview report → confirm with user → commit → create PR.

### Creating a new report type
1. clone_or_update()
2. git_branch("add-<report-name>-report")
3. Read an existing report (e.g. reports/executive.py) as a template
4. Create the new report file in reports/
5. Register it in reports/__init__.py
6. Export it in __init__.py
7. Add sample data to dataset.py if new data categories are needed
8. Update README.md with the new report type
9. run_python to verify the new report generates correctly
10. git_commit and create_pr

### Custom data from user
When a user pastes JSON or CSV data in chat:
1. Parse their data into the expected format (list of dicts)
2. Use generate_report_custom(report_type, json_string) to build the report
3. The UI shows preview/download buttons automatically

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
    """Code-maintenance agent for the mmm_report package."""

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
                    "- `read mmm_report/builder.py`\n"
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
