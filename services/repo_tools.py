"""repo_tools.py — File-system and git tools for managed external repos."""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent

REPOS = {
    "report": {
        "url": "https://github.com/Kuldeepsvts/report.git",
        "dir": APP_ROOT / "external_repos" / "report",
    },
}

# Backward-compatible module-level aliases for the default repo.
EXTERNAL_REPO_URL = REPOS["report"]["url"]
EXTERNAL_REPO_DIR = REPOS["report"]["dir"]


def _repo_entry(repo: str) -> dict:
    entry = REPOS.get(repo)
    if not entry:
        raise ValueError(f"Unknown repo: {repo!r}. Available: {', '.join(REPOS)}")
    return entry


def _repo_dir(repo: str) -> Path:
    return _repo_entry(repo)["dir"]


def _repo_url(repo: str) -> str:
    return _repo_entry(repo)["url"]


def _safe_path(relative_path: str, repo: str = "report") -> Path:
    repo_dir = _repo_dir(repo)
    cleaned = relative_path.lstrip("/").lstrip("\\")
    resolved = (repo_dir / cleaned).resolve()
    if not str(resolved).startswith(str(repo_dir.resolve())):
        raise ValueError(f"Path escapes repo directory: {relative_path}")
    return resolved


def _run_git(*cmd_args: str, cwd: Path = None, repo: str = "report") -> str:
    result = subprocess.run(
        ["git"] + list(cmd_args),
        cwd=str(cwd or _repo_dir(repo)),
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


def _ensure_cloned(repo: str = "report"):
    if not (_repo_dir(repo) / ".git").exists():
        raise RuntimeError("Repo not cloned yet. Call repo_clone_or_update first.")


def _default_branch(repo: str) -> str:
    """Resolve the remote's default branch (e.g. 'main') via origin/HEAD."""
    ref = _run_git("symbolic-ref", "refs/remotes/origin/HEAD", repo=repo)
    return ref.rsplit("/", 1)[-1]


def _repo_clone_or_update(args: dict) -> str:
    repo = args.get("repo", "report")
    repo_dir = _repo_dir(repo)
    repo_url = _repo_url(repo)
    if not (repo_dir / ".git").exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_git("clone", repo_url, str(repo_dir), cwd=repo_dir.parent, repo=repo)
        return f"Cloned {repo_url}"
    _run_git("fetch", "origin", repo=repo)
    default_branch = _default_branch(repo)
    current_branch = _run_git("rev-parse", "--abbrev-ref", "HEAD", repo=repo)
    if current_branch != default_branch:
        dirty = _run_git("status", "--porcelain", repo=repo)
        if dirty and dirty != "(no output)":
            raise RuntimeError(
                f"Currently on branch '{current_branch}' with uncommitted changes — "
                f"refusing to switch to '{default_branch}' and lose them. "
                f"Commit or discard the changes first (git_commit, or git_diff to review)."
            )
        _run_git("checkout", default_branch, repo=repo)
    return _run_git("pull", "origin", default_branch, repo=repo)


def _repo_status(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    return _run_git("status", "--short", repo=repo)


def _repo_diff(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    path = args.get("path")
    if path:
        safe = _safe_path(path, repo=repo)
        return _run_git("diff", "HEAD", "--", str(safe.relative_to(_repo_dir(repo))), repo=repo)
    return _run_git("diff", "HEAD", repo=repo)


def _repo_list_files(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    repo_dir = _repo_dir(repo)
    pattern = args.get("pattern", "")
    if pattern:
        matches = sorted(repo_dir.glob(pattern))
    else:
        matches = sorted(p for p in repo_dir.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in str(p))
    lines = [f"  {p.relative_to(repo_dir)}  ({p.stat().st_size} bytes)" for p in matches if ".git" not in p.parts]
    return f"{len(lines)} files:\n" + "\n".join(lines) if lines else "No files found"


def _repo_read_file(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    path = _safe_path(args["path"], repo=repo)
    if not path.exists(): return f"File not found: {args['path']}"
    content = path.read_text(encoding="utf-8", errors="replace")
    numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(content.splitlines())]
    return "\n".join(numbered)


def _repo_write_file(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    path = _safe_path(args["path"], repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return f"Wrote {args['content'].count(chr(10))+1} lines to {args['path']}"


def _repo_search_code(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    repo_dir = _repo_dir(repo)
    pattern = args["pattern"]
    file_pattern = args.get("file_pattern", "")
    files = sorted(repo_dir.glob(f"**/{file_pattern}")) if file_pattern else sorted(p for p in repo_dir.rglob("*") if p.is_file() and ".git" not in p.parts)
    try: regex = re.compile(pattern, re.IGNORECASE)
    except re.error: regex = re.compile(re.escape(pattern), re.IGNORECASE)
    results = []
    for fpath in files:
        try: content = fpath.read_text(encoding="utf-8", errors="replace")
        except: continue
        for i, line in enumerate(content.splitlines(), 1):
            if regex.search(line):
                results.append(f"  {fpath.relative_to(repo_dir)}:{i} | {line.rstrip()}")
    return f"{len(results)} matches:\n" + "\n".join(results[:80]) if results else f"No matches for '{pattern}'"


def _repo_branch(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    return _run_git("checkout", "-b", args["name"], repo=repo)


def _repo_checkout(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    return _run_git("checkout", args["name"], repo=repo)


def _repo_commit(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    _run_git("add", "-A", repo=repo)
    return _run_git("commit", "-m", args.get("message", "Update via agent"), repo=repo)


def _repo_push(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD", repo=repo)
    return _run_git("push", "-u", "origin", branch, repo=repo)


def _repo_create_pr(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    repo_dir = _repo_dir(repo)
    repo_url = _repo_url(repo)
    gh_repo_slug = repo_url.rstrip("/").removesuffix(".git").split("github.com/")[-1]
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD", repo=repo)
    if branch == "main": raise RuntimeError("On main — create a branch first.")
    _run_git("push", "-u", "origin", branch, repo=repo)
    result = subprocess.run(
        ["gh", "pr", "create", "--repo", gh_repo_slug, "--base", "main",
         "--head", branch, "--title", args.get("title", f"Agent: {branch}"),
         "--body", args.get("body", f"Automated PR (branch: {branch})")],
        cwd=str(repo_dir), capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0: raise RuntimeError((result.stdout + result.stderr).strip())
    return (result.stdout + result.stderr).strip()


def _repo_run_script(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    repo_dir = _repo_dir(repo)
    safe = _safe_path(args.get("script", ""), repo=repo)
    if not safe.exists(): raise FileNotFoundError(f"Not found: {args['script']}")
    extra_args = args.get("args", [])
    timeout = args.get("timeout", 30)
    r = subprocess.run(
        [sys.executable, str(safe), *extra_args],
        cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout,
    )
    return f"{'OK' if r.returncode==0 else f'FAILED({r.returncode})'}\n{r.stdout[:2000]}\n{r.stderr[:500]}".strip()


def _repo_run_python(args):
    repo = args.get("repo", "report")
    _ensure_cloned(repo)
    repo_dir = _repo_dir(repo)
    r = subprocess.run([sys.executable, "-c", args.get("code", "")], cwd=str(repo_dir),
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, "PYTHONPATH": str(repo_dir)})
    return f"{'OK' if r.returncode==0 else f'FAILED({r.returncode})'}\n{r.stdout[:2000]}\n{r.stderr[:500]}".strip()
