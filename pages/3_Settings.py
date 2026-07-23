"""
Settings — Configure LLM provider and display preferences.
"""

import sys
import json
from pathlib import Path

# Ensure backend is importable
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import streamlit as st

st.set_page_config(
    page_title="Settings — OOI RCA Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_utils.session import init_session_state
from streamlit_utils.ui_components import inject_custom_css, render_project_sidebar

init_session_state(st)
inject_custom_css()

# ── Sidebar ──
render_project_sidebar()

# ── Config file path ──
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(data: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Load current config ──
config = load_config()

st.markdown("# Settings")
st.markdown("---")


# ══════════════════════════════════════════════════════════════
# LLM Provider
# ══════════════════════════════════════════════════════════════

st.markdown("## LLM Provider")
st.markdown(
    "Works with any OpenAI-compatible endpoint — Fireworks (cloud) or a local "
    "server such as Ollama / LM Studio."
)

from core.config import settings as _settings

# ── Endpoint ──
api_base = st.text_input(
    "API Base URL",
    value=config.get("llm_api_base", "https://api.fireworks.ai/inference/v1"),
    key="llm_api_base_input",
    help="OpenAI-compatible base URL. Local Ollama: http://localhost:11434/v1",
)

def _set_endpoint(url, model):
    st.session_state.llm_api_base_input = url
    st.session_state.llm_model_input = model

col_fw, col_local = st.columns(2)
with col_fw:
    st.button("Use Fireworks (cloud)", key="preset_fw", on_click=_set_endpoint,
              args=("https://api.fireworks.ai/inference/v1", "accounts/fireworks/models/qwen3p7-plus"))
with col_local:
    st.button("Use Local Ollama", key="preset_local", on_click=_set_endpoint,
              args=("http://localhost:11434/v1", "qwen2.5:14b"))

# ── Model ──
llm_model = st.text_input(
    "Model",
    value=config.get("llm_model", "accounts/fireworks/models/qwen3p7-plus"),
    key="llm_model_input",
    help="Model id for the active endpoint. "
         "Fireworks: accounts/fireworks/models/...  ·  Ollama: qwen2.5:14b",
)

# ── API key (managed in secrets; local servers ignore it) ──
st.text_input(
    "API Key",
    value=_settings.LLM_API_KEY,
    type="password",
    disabled=True,
    key="llm_key_input",
    help="Managed via `.streamlit/secrets.toml`. Local servers (Ollama) ignore it.",
)

is_local = "localhost" in api_base or "127.0.0.1" in api_base
if not _settings.LLM_API_KEY and not is_local:
    st.warning(
        "No API key found in secrets. Add `FIREWORKS_API_KEY` (or `LLM_API_KEY`) "
        "to `.streamlit/secrets.toml`."
    )


st.markdown("---")


# ══════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════

st.markdown("## 🎨 Display")

display_name = st.text_input(
    "Display Name",
    value=config.get("display_name", ""),
    placeholder="Your name (shown in chat)",
    key="display_name_input",
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════
# Save Button
# ══════════════════════════════════════════════════════════════

if st.button("💾 Save Settings", type="primary", width='stretch'):
    provider_label = "local" if is_local else "fireworks"
    # Merge into existing config so unrelated fields are preserved.
    new_config = {
        **config,
        "display_name": display_name,
        "llm_provider": provider_label,
        "llm_api_base": api_base,
        "llm_model": llm_model,
    }

    save_config(new_config)

    # Hot-reload the LLM manager
    if st.session_state.llm_manager is not None:
        from core.config import settings

        st.session_state.llm_manager.set_provider(
            api_key=settings.LLM_API_KEY,
            model=llm_model,
            api_base=api_base,
            provider_name=provider_label,
        )

    st.success("Settings saved successfully!")
    st.info(f"Active endpoint: **{provider_label}** — `{llm_model}`  ·  {api_base}")


# ══════════════════════════════════════════════════════════════
# Status Footer
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("##### Current Status")

status_col1, status_col2 = st.columns(2)
with status_col1:
    st.metric("LLM Provider", config.get("llm_provider", "fireworks"))

with status_col2:
    _m = config.get("llm_model", "—")
    st.metric("Model", _m.split("/")[-1])
