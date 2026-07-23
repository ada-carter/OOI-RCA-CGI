import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR.parent / "projects"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def _load_secrets_toml() -> dict:
    """Load secrets.toml from the project root .streamlit dir."""
    secrets_path = BASE_DIR.parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
            with open(secrets_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.error(f"Error reading secrets.toml: {e}")
    return {}

def _load_config_json() -> dict:
    """Load config.json from the project root, returning empty dict on failure."""
    config_path = BASE_DIR.parent / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading config.json: {e}")
    return {}





class Settings:
    PROJECT_NAME: str = "OOI RCA Copilot"
    API_V1_STR: str = "/api/v1"

    @property
    def OOI_USERNAME(self) -> str:
        secrets = _load_secrets_toml()
        if secrets.get("OOI_USERNAME"):
            return secrets["OOI_USERNAME"]
        return os.environ.get("OOI_USERNAME", "")

    @property
    def OOI_TOKEN(self) -> str:
        secrets = _load_secrets_toml()
        if secrets.get("OOI_TOKEN"):
            return secrets["OOI_TOKEN"]
        return os.environ.get("OOI_TOKEN", "")

    # ── Fireworks AI / LLM Provider ──

    @property
    def LLM_PROVIDER(self) -> str:
        """Label for the active endpoint (e.g. 'fireworks', 'local'). Display only."""
        data = _load_config_json()
        return data.get("llm_provider", "fireworks")

    @property
    def LLM_API_BASE(self) -> str:
        """OpenAI-compatible base URL. Point at Fireworks, OpenAI, vLLM, Ollama, etc.

        Local example (Ollama): "http://localhost:11434/v1"
        """
        data = _load_config_json()
        return data.get("llm_api_base", "https://api.fireworks.ai/inference/v1")

    @property
    def LLM_MODEL(self) -> str:
        """Active model id (config.json `llm_model`)."""
        data = _load_config_json()
        return data.get("llm_model", "accounts/fireworks/models/llama-v3p3-70b-instruct")

    @property
    def LLM_API_KEY(self) -> str:
        """Key for the active endpoint. Local servers (Ollama, LM Studio) ignore it."""
        secrets = _load_secrets_toml()
        return (
            secrets.get("LLM_API_KEY")
            or secrets.get("FIREWORKS_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("FIREWORKS_API_KEY", "")
        )

    @property
    def DISPLAY_NAME(self) -> str:
        data = _load_config_json()
        return data.get("display_name", "") or "You"

    # ── Database ──
    @property
    def DATABASE_URL(self) -> str:
        secrets = _load_secrets_toml()
        if secrets.get("DATABASE_URL"):
            return secrets["DATABASE_URL"]
        return os.environ.get("DATABASE_URL", "")

    # ── Auth (Google OAuth) ──
    @property
    def GOOGLE_CLIENT_ID(self) -> str:
        secrets = _load_secrets_toml()
        if secrets.get("GOOGLE_CLIENT_ID"):
            return secrets["GOOGLE_CLIENT_ID"]
        return os.environ.get("GOOGLE_CLIENT_ID", "")

    @property
    def GOOGLE_CLIENT_SECRET(self) -> str:
        secrets = _load_secrets_toml()
        if secrets.get("GOOGLE_CLIENT_SECRET"):
            return secrets["GOOGLE_CLIENT_SECRET"]
        return os.environ.get("GOOGLE_CLIENT_SECRET", "")

    @property
    def OAUTH_REDIRECT_URI(self) -> str:
        secrets = _load_secrets_toml()
        if secrets.get("OAUTH_REDIRECT_URI"):
            return secrets["OAUTH_REDIRECT_URI"]
        return os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8501")

settings = Settings()
