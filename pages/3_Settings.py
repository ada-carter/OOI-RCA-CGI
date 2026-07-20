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
st.markdown("Choose between a local GGUF model (no internet required) or Fireworks AI cloud inference.")

provider = st.radio(
    "Provider",
    ["local", "fireworks"],
    index=0 if config.get("llm_provider", "local") == "local" else 1,
    format_func=lambda x: "Local (Gemma GGUF via llama.cpp)" if x == "local" else "Cloud (Fireworks AI)",
    horizontal=True,
    key="provider_radio",
)

st.markdown("")

if provider == "local":
    st.markdown("### Local Model Configuration")

    model_options = {
        "Gemma 4 12B Q8_0 (Best quality, ~12GB VRAM)": "gemma4-coding-Q8_0.gguf",
        "Gemma 4 12B Q4_K_M (Balanced, ~7GB VRAM)": "gemma4-coding-Q4_K_M.gguf",
        "Gemma 4 12B Q2_K (Smallest, ~4GB VRAM)": "gemma4-coding-Q2_K.gguf",
    }

    current_filename = config.get("model_filename", "gemma4-coding-Q8_0.gguf")
    current_idx = 0
    for i, (label, fname) in enumerate(model_options.items()):
        if fname == current_filename:
            current_idx = i
            break

    selected_model = st.selectbox(
        "Model",
        list(model_options.keys()),
        index=current_idx,
        key="local_model_select",
    )
    model_filename = model_options[selected_model]

    st.info(
        "Models are downloaded automatically from HuggingFace on first use. "
        "Ensure you have sufficient VRAM/RAM for the selected quantization."
    )

else:  # fireworks
    st.markdown("### Fireworks AI Configuration")

    from core.config import settings as _settings

    st.text_input(
        "API Key",
        value=_settings.FIREWORKS_API_KEY,
        type="password",
        disabled=True,
        key="fireworks_key_input",
        help="Managed via `.streamlit/secrets.toml` (or Streamlit Cloud Secrets dashboard).",
    )

    fireworks_model = st.text_input(
        "Model",
        value=config.get("fireworks_model", "accounts/fireworks/models/deepseek-v4-flash"),
        placeholder="accounts/fireworks/models/...",
        key="fireworks_model_input",
        help="Enter the full Fireworks model path. Common options: "
             "llama-v3p3-70b-instruct, qwen3-235b-a22b, deepseek-v4-flash",
    )

    st.markdown("##### Popular models")
    popular_models = {
        "DeepSeek V4 Flash": "accounts/fireworks/models/deepseek-v4-flash",
        "Llama 3.3 70B Instruct": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "Qwen 3 235B A22B": "accounts/fireworks/models/qwen3-235b-a22b",
    }
    
    def set_fw_model(m_id):
        st.session_state.fireworks_model_input = m_id

    for name, model_id in popular_models.items():
        st.button(f"Use {name}", key=f"use_{model_id}", on_click=set_fw_model, args=(model_id,))

    if not _settings.FIREWORKS_API_KEY:
        st.warning(
            "No API key found in secrets. Add `FIREWORKS_API_KEY` to `.streamlit/secrets.toml`."
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
    new_config = {
        "display_name": display_name,
        "llm_provider": provider,
        "model_repo_id": "yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF",
    }

    if provider == "local":
        new_config["model_filename"] = model_filename
        new_config["fireworks_model"] = config.get("fireworks_model", "accounts/fireworks/models/llama-v3p3-70b-instruct")
    else:
        new_config["model_filename"] = config.get("model_filename", "gemma4-coding-Q8_0.gguf")
        new_config["fireworks_model"] = fireworks_model

    save_config(new_config)

    # Hot-reload the LLM manager
    if st.session_state.llm_manager is not None:
        from core.config import settings, get_model_config

        st.session_state.llm_manager.set_provider(
            provider=provider,
            fireworks_api_key=settings.FIREWORKS_API_KEY,
            fireworks_model=new_config.get("fireworks_model", ""),
            model_config=get_model_config(),
        )

    st.success("Settings saved successfully!")

    # Show current provider status
    if provider == "fireworks":
        if settings.FIREWORKS_API_KEY:
            st.info(f"Active provider: Fireworks AI — `{new_config['fireworks_model']}`")
        else:
            st.warning("Fireworks AI selected but no API key found in secrets.")
    else:
        st.info(f"Active provider: Local — `{new_config['model_filename']}`")


# ══════════════════════════════════════════════════════════════
# Status Footer
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("##### Current Status")

status_col1, status_col2 = st.columns(2)
with status_col1:
    current_provider = config.get("llm_provider", "local")
    if config.get("llm_provider", "local") == "fireworks":
        st.metric("LLM Provider", "Fireworks AI")
    else:
        st.metric("LLM Provider", "Local")

with status_col2:
    if current_provider == "fireworks":
        st.metric("Model", config.get("fireworks_model", "—").split("/")[-1])
    else:
        st.metric("Model", config.get("model_filename", "—"))
