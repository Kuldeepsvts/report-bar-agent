# Nexus Agent

Standalone AI agent for maintaining the [mmm_report](https://github.com/Kuldeepsvts/report.git) Python package.

## Quick Start

```bash
cd nexus-agent
pip install -r requirements.txt
python app.py
```

Open http://localhost:5050 and login with `admin` / `admin123`.

## What It Does

- Generate 5 types of HTML reports (executive, sales, product, demographic, channel)
- Edit report code, preview changes instantly, then commit and create PRs
- Upload CSV/RDS files and generate reports from custom data
- Search code, run Python snippets, manage git branches
- Full conversation history with localStorage persistence

## Configuration

Edit `config.py`:

```python
AUTH_USERS = {"admin": "admin123"}     # Login credentials
CORTEX_API_BASE = "..."                # LLM API endpoint
LLM_MODEL = "bia-genai-llm-gpt-4o-128k"
REPO_URL = "https://github.com/Kuldeepsvts/report.git"
```

## Structure

```
nexus-agent/
├── app.py              — Flask app + routes + auth
├── config.py           — Configuration
├── requirements.txt    — Dependencies
├── services/
│   ├── agent_service.py — LangGraph agent (18 tools, RAG, retry logic)
│   └── repo_tools.py   — Git/file operations for the managed repo
├── templates/
│   └── agent.html      — Chat UI (3-panel, dark/light theme)
└── external_repos/     — Cloned report repo (created at runtime)
```

## 18 Agent Tools

| Tool | Description |
|------|-------------|
| `clone_or_update` | Clone or pull the report repo |
| `list_files` | List files with glob filter |
| `read_file` | Read file with line numbers |
| `write_file` | Create/overwrite files |
| `search_code` | Regex search across codebase |
| `git_status` | Show repo status |
| `git_diff` | Show changes |
| `git_branch` | Create new branch |
| `git_checkout` | Switch branches |
| `git_commit` | Stage and commit |
| `git_push` | Push to origin |
| `create_pr` | Create GitHub PR via `gh` CLI |
| `generate_report` | Build report from local code |
| `generate_report_custom` | Build report with custom JSON data |
| `list_report_types` | Show available report types |
| `run_script` | Execute Python script in repo |
| `run_python` | Run inline Python code |
| `get_uploaded_file` | Access uploaded CSV/RDS data |
