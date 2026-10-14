"""repo_tools.py — File-system and git tools for the managed external repo."""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_REPO_URL = "https://github.com/Kuldeepsvts/report.git"
EXTERNAL_REPO_DIR = APP_ROOT / "external_repos" / "report"


def _safe_path(relative_path: str) -> Path:
    cleaned = relative_path.lstrip("/").lstrip("\\")
    resolved = (EXTERNAL_REPO_DIR / cleaned).resolve()
    if not str(resolved).startswith(str(EXTERNAL_REPO_DIR.resolve())):
        raise ValueError(f"Path escapes repo directory: {relative_path}")
    return resolved


def _run_git(*cmd_args: str, cwd: Path = None) -> str:
    result = subprocess.run(
        ["git"] + list(cmd_args),
        cwd=str(cwd or EXTERNAL_REPO_DIR),
        capture_output=True, text=True, timeout=60,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"git {' '.join(cmd_args)} failed")
    return output or "(no output)"


def execute_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    handlers = {
        "repo_clone_or_update": _repo_clone_or_update,
        "repo_status": _repo_status,
        "repo_diff": _repo_diff,
        "repo_list_files": _repo_list_files,
        "repo_read_file": _repo_read_file,
        "repo_write_file": _repo_write_file,
        "repo_search_code": _repo_search_code,
        "repo_branch": _repo_branch,
        "repo_checkout": _repo_checkout,
        "repo_commit": _repo_commit,
        "repo_push": _repo_push,
        "repo_create_pr": _repo_create_pr,
        "repo_run_script": _repo_run_script,
        "repo_run_python": _repo_run_python,
    }
    handler = handlers.get(tool_name)
    if not handler:
        return {"success": False, "error": f"Unknown tool: {tool_name}"}
    try:
        result = handler(args)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _ensure_cloned():
    if not (EXTERNAL_REPO_DIR / ".git").exists():
        raise RuntimeError("Repo not cloned yet. Call repo_clone_or_update first.")


def _repo_clone_or_update(args: dict) -> str:
    if not (EXTERNAL_REPO_DIR / ".git").exists():
        EXTERNAL_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run_git("clone", EXTERNAL_REPO_URL, str(EXTERNAL_REPO_DIR), cwd=EXTERNAL_REPO_DIR.parent)
        return f"Cloned {EXTERNAL_REPO_URL}"
    _run_git("fetch", "origin")
    return _run_git("pull")


def _repo_status(args): _ensure_cloned(); return _run_git("status", "--short")

def _repo_diff(args):
    _ensure_cloned()
    path = args.get("path")
    if path:
        safe = _safe_path(path)
        return _run_git("diff", "HEAD", "--", str(safe.relative_to(EXTERNAL_REPO_DIR)))
    return _run_git("diff", "HEAD")

def _repo_list_files(args):
    _ensure_cloned()
    pattern = args.get("pattern", "")
    if pattern:
        matches = sorted(EXTERNAL_REPO_DIR.glob(pattern))
    else:
        matches = sorted(p for p in EXTERNAL_REPO_DIR.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in str(p))
    lines = [f"  {p.relative_to(EXTERNAL_REPO_DIR)}  ({p.stat().st_size} bytes)" for p in matches if ".git" not in p.parts]
    return f"{len(lines)} files:\n" + "\n".join(lines) if lines else "No files found"

def _repo_read_file(args):
    _ensure_cloned()
    path = _safe_path(args["path"])
    if not path.exists(): return f"File not found: {args['path']}"
    content = path.read_text(encoding="utf-8", errors="replace")
    numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(content.splitlines())]
    return "\n".join(numbered)

def _repo_write_file(args):
    _ensure_cloned()
    path = _safe_path(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return f"Wrote {args['content'].count(chr(10))+1} lines to {args['path']}"

def _repo_search_code(args):
    _ensure_cloned()
    pattern = args["pattern"]
    file_pattern = args.get("file_pattern", "")
    files = sorted(EXTERNAL_REPO_DIR.glob(f"**/{file_pattern}")) if file_pattern else sorted(p for p in EXTERNAL_REPO_DIR.rglob("*") if p.is_file() and ".git" not in p.parts)
    try: regex = re.compile(pattern, re.IGNORECASE)
    except re.error: regex = re.compile(re.escape(pattern), re.IGNORECASE)
    results = []
    for fpath in files:
        try: content = fpath.read_text(encoding="utf-8", errors="replace")
        except: continue
        for i, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                results.append(f"  {fpath.relative_to(EXTERNAL_REPO_DIR)}:{i} | {line.rstrip()}")
    return f"{len(results)} matches:\n" + "\n".join(results[:80]) if results else f"No matches for '{pattern}'"

def _repo_branch(args): _ensure_cloned(); return _run_git("checkout", "-b", args["name"])
def _repo_checkout(args): _ensure_cloned(); return _run_git("checkout", args["name"])

def _repo_commit(args):
    _ensure_cloned()
    _run_git("add", "-A")
    return _run_git("commit", "-m", args.get("message", "Update via agent"))

def _repo_push(args):
    _ensure_cloned()
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return _run_git("push", "-u", "origin", branch)

def _repo_create_pr(args):
    _ensure_cloned()
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "main": raise RuntimeError("On main — create a branch first.")
    _run_git("push", "-u", "origin", branch)
    result = subprocess.run(
        ["gh", "pr", "create", "--repo", "Kuldeepsvts/report", "--base", "main",
         "--head", branch, "--title", args.get("title", f"Agent: {branch}"),
         "--body", args.get("body", f"Automated PR (branch: {branch})")],
        cwd=str(EXTERNAL_REPO_DIR), capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0: raise RuntimeError((result.stdout + result.stderr).strip())
    return (result.stdout + result.stderr).strip()

def _repo_run_script(args):
    _ensure_cloned()
    safe = _safe_path(args.get("script", ""))
    if not safe.exists(): raise FileNotFoundError(f"Not found: {args['script']}")
    r = subprocess.run([sys.executable, str(safe)], cwd=str(EXTERNAL_REPO_DIR), capture_output=True, text=True, timeout=30)
    return f"{'OK' if r.returncode==0 else f'FAILED({r.returncode})'}\n{r.stdout[:2000]}\n{r.stderr[:500]}".strip()

def _repo_run_python(args):
    _ensure_cloned()
    r = subprocess.run([sys.executable, "-c", args.get("code", "")], cwd=str(EXTERNAL_REPO_DIR),
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, "PYTHONPATH": str(EXTERNAL_REPO_DIR)})
    return f"{'OK' if r.returncode==0 else f'FAILED({r.returncode})'}\n{r.stdout[:2000]}\n{r.stderr[:500]}".strip()
