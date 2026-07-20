"""
AIDA — OOI RCA Copilot
Streamlit application entry point.

Redirects to the Chat page as the default landing page.
"""

import sys
from pathlib import Path

# Ensure backend is importable from all pages
_BACKEND_DIR = str(Path(__file__).resolve().parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import streamlit as st

st.set_page_config(
    page_title="OOI RCA Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_utils.auth import require_auth, render_login_page
from streamlit_utils.session import init_session_state
from streamlit_utils.ui_components import inject_custom_css, render_project_sidebar

init_session_state(st)
inject_custom_css()

if not require_auth():
    render_login_page()
    st.stop()

# ── Sidebar ──
render_project_sidebar()

# ── Main content — landing page ──
st.markdown(f"""
# OOI RCA Copilot

Welcome to the **Ocean Observatories Initiative Regional Cabled Array Copilot**, {st.session_state.user_name}.

---

### What can I do?

| Capability | Description |
|---|---|
| **Chat** | Ask questions, explore instruments, request data downloads |
| **M2M API** | Automatically query the OOI Machine-to-Machine API |
| **Data Processing** | Concatenate NetCDF files, compute statistics, generate plots |
| **Script Generation** | Get reproducible Python scripts for your analysis |
| **RAG Knowledge** | Leverage built-in RCA instrumentation documentation |

### Get Started

Use the sidebar to start a new chat, or navigate to the **Chat** page to jump right in.

---
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/1_Chat.py", label="Start Chatting")
with col2:
    st.page_link("pages/2_Data.py", label="View Data")
with col3:
    st.page_link("pages/3_Settings.py", label="Settings")

# Provider status
from streamlit_utils.session import get_llm_manager
try:
    mgr = get_llm_manager(st)
    provider = mgr.provider_name
    if provider == "fireworks":
        st.success("Connected to **Fireworks AI** cloud LLM")
    else:
        if mgr.model is not None:
            st.success("Local LLM loaded and ready (**Gemma GGUF**)")
        else:
            st.warning("Local LLM provider selected but model not loaded. Check Settings.")
except Exception as e:
    st.error(f"LLM initialization error: {e}")
