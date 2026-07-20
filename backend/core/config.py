import os
from pathlib import Path
import logging
from huggingface_hub import hf_hub_download

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


def get_model_config() -> dict:
    data = _load_config_json()
    return {
        "repo_id": data.get("model_repo_id", "yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF"),
        "filename": data.get("model_filename", "gemma4-coding-Q8_0.gguf"),
    }


class Settings:
    PROJECT_NAME: str = "OOI RCA Copilot"
    API_V1_STR: str = "/api/v1"

    # Model config
    @property
    def MODEL_CONFIG(self) -> dict:
        return get_model_config()

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
        """'local' (llama.cpp) or 'fireworks' (Fireworks AI cloud)."""
        data = _load_config_json()
        return data.get("llm_provider", "local") or os.environ.get("LLM_PROVIDER", "local")

    @property
    def FIREWORKS_API_KEY(self) -> str:
        secrets = _load_secrets_toml()
        if secrets.get("FIREWORKS_API_KEY"):
            return secrets["FIREWORKS_API_KEY"]
        return os.environ.get("FIREWORKS_API_KEY", "")

    @property
    def FIREWORKS_MODEL(self) -> str:
        data = _load_config_json()
        return data.get("fireworks_model", "accounts/fireworks/models/llama-v3p3-70b-instruct")

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
