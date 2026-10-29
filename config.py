"""Nexus Agent — Configuration."""

SECRET_KEY = "nexus-agent-dev-key-change-in-production"

CORTEX_API_BASE = "https://api.cortex.lilly.com/cortex-openai"
# LLM_MODEL = "bia-genai-llm-gpt-4o-128k"
LLM_MODEL = "report-agent"
# EMBEDDING_MODEL = "bia-genai-embedding-gpt-small"
EMBEDDING_MODEL = "report-agent-embedding"

REPO_URL = "https://github.com/EliLillyCo/report.git" #"https://github.com/Kuldeepsvts/report.git"
# mmm_report_platform now lives inside REPO_URL's repo as a sibling package
# (mmm_report_platform/), not a separate repo.

AUTH_USERS = {
    "admin": "admin123",
    "kuldeep": "kuldeep123",
}

