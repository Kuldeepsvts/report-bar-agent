"""Nexus Agent — Configuration."""

SECRET_KEY = "nexus-agent-dev-key-change-in-production"

CORTEX_API_BASE = "https://api.cortex.lilly.com/cortex-openai"
# LLM_MODEL = "bia-genai-llm-gpt-4o-128k"
LLM_MODEL = "report-agent"
# EMBEDDING_MODEL = "bia-genai-embedding-gpt-small"
EMBEDDING_MODEL = "report-agent-embedding"

REPO_URL = "https://github.com/Kuldeepsvts/report.git"

AUTH_USERS = {
    "admin": "admin123",
    "kuldeep": "kuldeep123",
}
