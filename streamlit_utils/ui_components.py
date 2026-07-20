"""
Reusable Streamlit UI components for the AIDA app.
"""

import streamlit as st
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Custom CSS — injected once per page render
# ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Global font & color refinements ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Streamlit header bar & nav ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stSidebarNav"] {display: none;}
.stAppDeployButton {display: none;}

/* ── Sidebar styling (Glassmorphism) ── */
section[data-testid="stSidebar"] {
    background-color: rgba(20, 20, 20, 0.6) !important;
    backdrop-filter: blur(16px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    border-radius: 0px !important;
    padding: 8px 14px;
    font-size: 13px;
    font-weight: 500;
    transition: background-color 0.15s;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e5e5e5 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255, 255, 255, 0.05);
    color: white !important;
}

/* ── Chat message styling & Accents ── */
[data-testid="stChatMessage"] {
    padding: 12px 16px;
    border-radius: 0px !important;
    max-width: 100%;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.05);
}

/* User Accent (Green Crab) */
[data-testid="stChatMessage"]:has(.msg-user) {
    background-color: rgba(34, 197, 94, 0.05) !important;
    border-left: 3px solid rgba(34, 197, 94, 0.4) !important;
}

/* Assistant Accent (Blue Fish) */
[data-testid="stChatMessage"]:has(.msg-assistant) {
    background-color: rgba(56, 189, 248, 0.05) !important;
    border-left: 3px solid rgba(56, 189, 248, 0.4) !important;
}

/* ── Chat input styling ── */
[data-testid="stChatInput"] textarea {
    font-size: 14px !important;
    border-radius: 0px !important;
}
[data-testid="stChatInput"] {
    border-radius: 0px !important;
    background: rgba(30,30,30,0.5);
    backdrop-filter: blur(12px);
}

/* ── M2M approval card (Glassmorphism) ── */
.m2m-approval-card {
    background-color: rgba(30, 30, 30, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0px !important;
    padding: 16px 20px;
    margin: 8px 0;
}
.m2m-approval-card h4 {
    color: #3b82f6;
    margin-bottom: 8px;
}
.m2m-approval-card button {
    border-radius: 0px !important;
}

/* ── Pipeline flowchart (mermaid) ── */
.stMarkdown .element-container {
    overflow-x: auto;
}

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 500;
    border-radius: 0px !important;
}

/* ── Data table adjustments ── */
[data-testid="stDataFrame"] {
    border-radius: 0px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: rgba(30, 30, 30, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0px !important;
    padding: 12px 16px;
}

/* ── Expander (for thoughts) ── */
details {
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 0px !important;
    padding: 4px 12px;
    margin-bottom: 8px;
    background: rgba(20,20,20,0.4);
}
details summary {
    color: #808080;
    font-size: 12px;
    cursor: pointer;
}

/* ── Thinking Animation Gradient ── */
@keyframes pulseGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.thinking-block {
    background: linear-gradient(90deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1), rgba(16,185,129,0.1));
    background-size: 200% 200%;
    animation: pulseGradient 3s ease infinite;
    padding: 10px;
    border-left: 3px solid #10b981;
    margin-bottom: 10px;
    font-family: monospace;
    font-size: 13px;
    color: #a3a3a3;
}
</style>
"""


def inject_custom_css():
    """Inject custom CSS into the Streamlit page. Call once per page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Project Sidebar
# ──────────────────────────────────────────────────────────────

def render_project_sidebar():
    """
    Render the project sidebar with create/select/delete functionality.
    Returns the currently selected project_id (or None).
    """
    import sys
    from pathlib import Path
    _backend = str(Path(__file__).resolve().parent.parent / "backend")
    if _backend not in sys.path:
        sys.path.insert(0, _backend)
    from state.project_manager import project_manager

    with st.sidebar:
        # User Profile Badge
        if "user_name" in st.session_state:
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.session_state.get("user_picture"):
                    st.image(st.session_state.user_picture, width=40)
                else:
                    st.markdown("👤")
            with c2:
                st.markdown(f"**{st.session_state.user_name}**")
                st.caption(st.session_state.user_email)
            
            if st.button("Logout", key="logout_btn", width='stretch'):
                st.session_state.clear()
                st.rerun()
                
            st.markdown("---")

        st.image("assets/loading_grenadier.svg", width='stretch')
        st.markdown("---")

        # New chat button
        if st.button("+ New Chat", key="new_chat_btn", width='stretch'):
            st.session_state.current_project = None
            st.session_state.messages = []
            st.session_state.active_plot = None
            st.session_state.active_csv = None
            st.session_state.active_flowchart = None
            try:
                st.switch_page("pages/1_Chat.py")
            except Exception:
                st.rerun()

        st.markdown("##### Recent")

        if "user_id" not in st.session_state:
            return None

        projects = project_manager.list_projects(st.session_state.user_id)
        if not projects:
            st.caption("No chats yet. Start chatting to create one!")

        for p in projects:
            proj_id = p["id"]
            proj_title = p["title"]
            col1, col2 = st.columns([5, 1])
            with col1:
                is_active = st.session_state.current_project == proj_id
                label = f"**{proj_title}**" if is_active else proj_title
                if st.button(
                    label,
                    key=f"proj_{proj_id}",
                    width='stretch',
                    type="primary" if is_active else "secondary",
                ):
                    if st.session_state.current_project != proj_id:
                        st.session_state.current_project = proj_id
                        project_manager.current_project_id = proj_id
                        # Load chat history
                        history = project_manager.load_chat_history(proj_id)
                        st.session_state.messages = history
                        # Load view state
                        st.session_state.active_flowchart = project_manager.load_flowchart(proj_id)
                        
                    try:
                        st.switch_page("pages/1_Chat.py")
                    except Exception:
                        st.rerun()
            with col2:
                if st.button("X", key=f"del_{proj_id}", help=f"Delete {proj_title}"):
                    project_manager.delete_project(proj_id)
                    if st.session_state.current_project == proj_id:
                        st.session_state.current_project = None
                        st.session_state.messages = []
                    st.rerun()

            if is_active:
                purpose = p["purpose"]
                if purpose:
                    st.caption(f"**Objective:** {purpose}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.page_link("pages/2_Data.py", label="📊 Data")
                with c2:
                    st.page_link("pages/4_Analysis.py", label="🔬 Analysis")

        # Bottom links
        st.markdown("---")
        st.page_link("pages/3_Settings.py", label="Settings")

    return st.session_state.current_project


# ──────────────────────────────────────────────────────────────
# M2M Approval Card
# ──────────────────────────────────────────────────────────────

def render_m2m_approval(params: dict, approval_key: str) -> str:
    """
    Render an M2M approval card with Accept/Reject buttons.
    Returns "accepted", "rejected", or "pending".
    """
    state_key = f"m2m_approval_{approval_key}"

    if state_key in st.session_state and st.session_state[state_key] != "pending":
        status = st.session_state[state_key]
        if status == "accepted":
            st.success("M2M request approved — submitting to OOI API...")
        else:
            st.error("M2M request rejected.")
        return status

    st.markdown('<div class="m2m-approval-card">', unsafe_allow_html=True)
    st.markdown("#### M2M Data Download Request")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Subsite:** `{params.get('subsite', '?')}`")
        st.markdown(f"**Node:** `{params.get('node', '?')}`")
        st.markdown(f"**Sensor:** `{params.get('sensor', '?')}`")
    with col2:
        st.markdown(f"**Method:** `{params.get('method', '?')}`")
        st.markdown(f"**Stream:** `{params.get('stream', '?')}`")
        st.markdown(f"**Time:** `{params.get('begin_dt', '?')}` -> `{params.get('end_dt', '?')}`")

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("Accept", key=f"accept_{approval_key}", type="primary"):
            st.session_state[state_key] = "accepted"
            st.rerun()
    with btn_col2:
        if st.button("Reject", key=f"reject_{approval_key}"):
            st.session_state[state_key] = "rejected"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    return "pending"


# ──────────────────────────────────────────────────────────────
# Display Helpers
# ──────────────────────────────────────────────────────────────

def render_flowchart_mermaid(steps_str: str):
    """Render a pipeline flowchart using Mermaid syntax."""
    if not steps_str:
        st.info("Pipeline steps will appear here after the agent processes data.")
        return

    steps = [s.strip() for s in steps_str.split("->") if s.strip()]
    if not steps:
        return

    # Build mermaid graph
    mermaid_lines = ["graph LR"]
    for i, step in enumerate(steps):
        node_id = f"N{i}"
        safe_label = step.replace('"', "'")
        mermaid_lines.append(f'    {node_id}["{safe_label}"]')
        if i > 0:
            prev_id = f"N{i-1}"
            mermaid_lines.append(f"    {prev_id} --> {node_id}")

    # Style nodes
    mermaid_lines.append("    classDef default fill:#2563eb,stroke:#1d4ed8,color:#fff,stroke-width:2px")

    mermaid_code = "\n".join(mermaid_lines)
    st.markdown(f"```mermaid\n{mermaid_code}\n```")
