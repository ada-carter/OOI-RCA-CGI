"""
Session state initialization and helpers for the Streamlit app.

Centralizes all session state keys so pages can safely access them
without guarding against KeyError on every read.
"""

import sys
from pathlib import Path

# Ensure backend is importable
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def init_session_state(st):
    """Initialize all session-state keys with sensible defaults.

    Call this at the top of every page to guarantee keys exist.
    """
    defaults = {
        # Chat
        "messages": [],            # list of {"role": "user"|"assistant", "content": str}
        "current_project": None,   # active project id (str)
        "streaming": False,        # True while LLM is generating

        # LLM
        "llm_manager": None,       # LLMManager instance (lazily created)

        # M2M approval
        "m2m_pending": [],         # list of pending approval dicts

        # Data view
        "active_plot": None,       # path to current plot image
        "active_csv": None,        # path to current CSV
        "active_flowchart": "",    # flowchart step string ("Download -> Clean -> Plot")

        # UI
        "theme": "dark",
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_llm_manager(st):
    """Lazily create or return the cached LLMManager instance."""
    if st.session_state.llm_manager is None:
        from core.config import settings
        from core.llm_manager import LLMManager

        st.session_state.llm_manager = LLMManager(
            fireworks_api_key=settings.FIREWORKS_API_KEY,
            fireworks_model=settings.FIREWORKS_MODEL,
        )
    return st.session_state.llm_manager
