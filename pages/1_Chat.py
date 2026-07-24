"""
Chat — The main conversational interface for OOI RCA Copilot.

Supports streaming LLM responses, instrument search, M2M data requests
with inline approval, and automatic project creation.
"""

import sys
import re
import uuid
from pathlib import Path

# Ensure backend is importable
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import streamlit as st

st.set_page_config(
    page_title="Chat — OOI RCA Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_utils.session import init_session_state, get_llm_manager
from streamlit_utils.ui_components import (
    inject_custom_css,
    render_project_sidebar,
    render_m2m_approval,
)
from streamlit_utils.chat_engine import (
    agent_loop,
    agent_loop_with_m2m_continuation,
    parse_thoughts,
    parse_update_view,
)

init_session_state(st)
inject_custom_css()

# ── Sidebar ──
render_project_sidebar()

from state.project_manager import project_manager
from controllers.chat_controller import ChatController
from core.config import settings

# ──────────────────────────────────────────────────────────────
# Helper: clean display text
# ──────────────────────────────────────────────────────────────

def clean_for_display(text: str) -> tuple[str, list[tuple[str, str]], list[dict]]:
    """Strip internal tags and format thoughts for display. Returns (display_text, list_of_plot_tuples, list_of_map_dicts)."""
    thoughts, response = parse_thoughts(text)

    # Extract plot paths before stripping system tags
    import re
    # Handle the new format with Dataset, fallback to old format just in case
    plots = re.findall(r'\[System: Plot saved to (.*?) \| Dataset: (.*?)\]', response)
    old_plots = re.findall(r'\[System: Plot saved to ([^|]*?)\]', response)
    
    for p in old_plots:
        plots.append((p.strip(), ""))
        
    maps_extracted = re.findall(r'\[System: Map rendered at lat=(.*?), lon=(.*?), title=(.*?)\]', response)
    maps = [{"lat": float(lat), "lon": float(lon), "title": title} for lat, lon, title in maps_extracted]

    # Strip m2m tags
    response = re.sub(r'<m2m_request\s+[^>]*/\s*>', '', response)
    # Strip zarr tags
    response = re.sub(r'<zarr_request\s+[^>]*/\s*>', '', response)
    # Strip update_view tags
    response = re.sub(r'<update_view\s+[^/>]+/?>', '', response)
    # Strip search_instruments tags
    response = re.sub(r'<search_instruments\s+[^>]*/\s*>', '', response)
    # Strip render_map tags (just in case they slipped through)
    response = re.sub(r'<render_map\s+[^>]*/\s*>', '', response)
    # Strip system messages
    response = re.sub(r'\[System:.*?\]', '', response)

    response = response.strip()

    if thoughts:
        formatted = f"""<details class="thinking-block" style="margin-bottom: 1rem; padding: 0.5rem; border-radius: 4px; border: 1px solid rgba(255,255,255,0.1); background-color: rgba(255,255,255,0.05);">
  <summary style="cursor: pointer; font-weight: 500; color: #aaa;">💭 Model Thoughts</summary>
  <div style="margin-top: 0.8rem; font-size: 0.9em; color: #ccc; white-space: pre-wrap;">{thoughts}</div>
</details>

{response}"""
    else:
        formatted = response
        
    return formatted, plots, maps

# ──────────────────────────────────────────────────────────────
# Global Gallery Dialog
# ──────────────────────────────────────────────────────────────
@st.dialog("🖼️ Plot Gallery")
def plot_gallery_dialog(pid: str):
    from pathlib import Path
    plot_folder = Path(f"projects/{pid}/plots")
    if plot_folder.exists():
        plots = list(plot_folder.glob("*.png"))
        if plots:
            for p in sorted(plots, key=lambda x: x.stat().st_mtime, reverse=True):
                st.image(str(p))
                with open(p, "rb") as f:
                    st.download_button("Download", data=f, file_name=p.name, mime="image/png", key=f"dl_gal_{p.name}_{uuid.uuid4().hex[:4]}")
                st.markdown("---")
        else:
            st.info("No plots found in this chat yet.")
    else:
        st.info("No plots found in this chat yet.")

# ──────────────────────────────────────────────────────────────
# Render existing messages
# ──────────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg.get("content", "") or msg.get("text", "")
    pid = st.session_state.current_project or "default"

    # Map old format roles
    if role == "copilot":
        role = "assistant"

    with st.chat_message(role, avatar="🦀" if role == "user" else "🐟"):
        if role == "assistant":
            display, plots, maps = clean_for_display(content)
            st.markdown(f"<div class='msg-assistant'></div>\n{display}", unsafe_allow_html=True)
            
            # Re-render any plots associated with this message
            for plot_idx, (plot_path, csv_str) in enumerate(plots):
                from pathlib import Path
                if Path(plot_path).exists():
                    with open(plot_path, "rb") as img_f:
                        st.image(img_f.read())
                    col1, col2, col3 = st.columns([1, 1, 1])
                    plot_name = Path(plot_path).name
                    with col1:
                        with open(plot_path, "rb") as f:
                            st.download_button("💾 Download Plot", data=f, file_name=plot_name, mime="image/png", key=f"dl_hist_{i}_{plot_name}")
                    with col2:
                        if st.button("🖼️ View Gallery", key=f"gal_hist_{i}_{plot_name}"):
                            plot_gallery_dialog(pid)
                    with col3:
                        if not csv_str:
                            base_proc = Path(plot_path).parent.parent / "processed_data"
                            if (base_proc / "processed.nc").exists():
                                csv_str = str(base_proc / "processed.nc")
                            else:
                                csv_str = str(base_proc / "processed.csv")
                            
                        csv_list = [c.strip() for c in csv_str.split(",")]
                        for idx, csv_path_str in enumerate(csv_list):
                            if Path(csv_path_str).exists():
                                with open(csv_path_str, "rb") as f:
                                    label = "📥 Download Dataset" if len(csv_list) == 1 else f"📥 Download Dataset {idx+1}"
                                    st.download_button(label, data=f, file_name=Path(csv_path_str).name, mime="text/csv", key=f"dl_csv_hist_{i}_{plot_name}_{idx}")
                    
                    from streamlit_utils.chat_engine import parse_plot_requests
                    plot_requests = parse_plot_requests(content)
                    
                    if plot_idx < len(plot_requests):
                        orig_params = plot_requests[plot_idx]
                        orig_params["csv"] = csv_str
                        with st.expander("🎛️ Tweak Plot"):
                            with st.form(key=f"tweak_form_{i}_{plot_idx}"):
                                t_title = st.text_input("Title", value=orig_params.get("title", f"{orig_params.get('y_col')} vs {orig_params.get('x_col')}"))
                                c1, c2 = st.columns(2)
                                with c1:
                                    t_xmin = st.text_input("X Min", value=orig_params.get("x_min", ""), placeholder="e.g. 10")
                                    t_xmax = st.text_input("X Max", value=orig_params.get("x_max", ""), placeholder="e.g. 100")
                                with c2:
                                    t_ymin = st.text_input("Y Min", value=orig_params.get("y_min", ""), placeholder="e.g. 0")
                                    t_ymax = st.text_input("Y Max", value=orig_params.get("y_max", ""), placeholder="e.g. 50")
                                
                                if st.form_submit_button("Update Plot"):
                                    orig_params["title"] = t_title
                                    orig_params["x_min"] = t_xmin if t_xmin else None
                                    orig_params["x_max"] = t_xmax if t_xmax else None
                                    orig_params["y_min"] = t_ymin if t_ymin else None
                                    orig_params["y_max"] = t_ymax if t_ymax else None
                                    orig_params["force_plot_path"] = plot_path
                                    
                                    from streamlit_utils.plot_renderer import generate_and_save_plot
                                    generate_and_save_plot(orig_params)
                                    st.rerun()
            
            # Re-render any maps associated with this message
            for m in maps:
                lat, lon = float(m["lat"]), float(m["lon"])
                bbox = f"{lon-0.5}%2C{lat-0.5}%2C{lon+0.5}%2C{lat+0.5}"
                osm_html = f'''
                <div style="margin-top: 1rem; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.2);">
                    <div style="background-color: rgba(255,255,255,0.1); padding: 0.5rem 1rem; font-weight: 500;">📍 {m["title"]}</div>
                    <iframe width="100%" height="350" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" 
                    src="https://www.openstreetmap.org/export/embed.html?bbox={bbox}&amp;layer=mapnik&amp;marker={lat}%2C{lon}">
                    </iframe>
                </div>
                '''
                st.markdown(osm_html, unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg-user'></div>\n{content}", unsafe_allow_html=True)

    # Check for pending M2M approval in this message
    if role == "assistant" and msg.get("m2m_pending"):
        params = msg["m2m_pending"]
        approval_key = msg.get("m2m_approval_key", f"m2m_{i}")
        status = render_m2m_approval(params, approval_key)

        if status == "accepted" and not msg.get("m2m_handled"):
            msg["m2m_handled"] = True
            # Execute M2M request and auto-process data
            with st.spinner("Submitting request to OOI M2M API..."):
                try:
                    from services.m2m_client import M2MClient

                    client = M2MClient(settings.OOI_USERNAME, settings.OOI_TOKEN)
                    pid = st.session_state.current_project or "default"
                    raw_dir = str(project_manager.get_raw_data_dir(pid))
                    processed_dir = str(project_manager.get_processed_data_dir(pid))

                    response = client.request_data(
                        subsite=params["subsite"],
                        node=params["node"],
                        sensor=params["sensor"],
                        method=params["method"],
                        stream=params["stream"],
                        begin_dt=params["begin_dt"],
                        end_dt=params["end_dt"],
                    )
                    request_uuid = response.get("requestUUID", "unknown")
                    thredds_url = client.get_thredds_url(response) or "not available"

                    obs = (
                        f"M2M request submitted successfully.\n"
                        f"Request UUID: {request_uuid}\n"
                        f"THREDDS URL: {thredds_url}\n"
                        f"Raw data directory: {raw_dir}\n"
                        f"Processed data directory: {processed_dir}\n"
                        f"Tell the user: their data request was submitted with UUID {request_uuid}. "
                        f"Data will be available on THREDDS once the server processes the request. "
                        f"Use <update_view> to update the UI when appropriate."
                    )
                    st.success(f"Request submitted! UUID: `{request_uuid}`")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "thredds_import",
                        "thredds_url": thredds_url,
                        "raw_dir": raw_dir,
                        "processed_dir": processed_dir,
                        "imported": False
                    })
                except Exception as e:
                    obs = f"M2M request failed with error: {str(e)}"
                    st.error(f"M2M request failed: {e}")

            # Continue agent loop with observation
            llm_mgr = get_llm_manager(st)
            pid = st.session_state.current_project or "default"
            original_msg = st.session_state.messages[0]["content"] if st.session_state.messages else ""

            accumulated = ""
            with st.chat_message("assistant", avatar="🐟"):
                placeholder = st.empty()
                for event in agent_loop_with_m2m_continuation(
                    message=original_msg, 
                    project_id=pid, 
                    llm_manager=llm_mgr, 
                    m2m_observation=obs, 
                    prior_accumulated=content,
                    session_messages=st.session_state.messages
                ):
                    if event["type"] == "token":
                        accumulated += event["text"]
                        display, _, _ = clean_for_display(accumulated)
                        placeholder.markdown(f"<div class='msg-assistant'></div>\n{display}", unsafe_allow_html=True)
                    elif event["type"] == "system":
                        st.info(f"{event['text']}")
                    elif event["type"] == "search_results":
                        with st.expander("🔍 Instrument Search Results", expanded=False):
                            st.code(event["text"], language=None)
                    elif event["type"] == "update_view":
                        params = event["params"]
                        if "plot" in params:
                            st.session_state.active_plot = params["plot"]
                        if "dataset" in params:
                            st.session_state.active_dataset = params["dataset"]
                        if "flowchart" in params:
                            st.session_state.active_flowchart = params["flowchart"]
                            project_manager.save_flowchart(pid, params["flowchart"])

                    elif event["type"] == "generate_plot":
                        params = event["params"]
                        dataset_file = params["dataset"]
                        
                        try:
                            from streamlit_utils.plot_renderer import generate_and_save_plot
                            from pathlib import Path
                            
                            # We can force the plot_path so we know where it ends up, or let the renderer decide
                            plot_dir = Path(dataset_file.split(",")[0].strip()).parent.parent / "plots"
                            plot_dir.mkdir(parents=True, exist_ok=True)
                            import uuid
                            plot_name = f"plot_{uuid.uuid4().hex[:8]}.png"
                            plot_path = plot_dir / plot_name
                            
                            params["force_plot_path"] = str(plot_path)
                            
                            plot_path_str = generate_and_save_plot(params)
                            
                            if plot_path_str:
                                st.session_state.active_plot = plot_path_str
                                st.image(plot_path_str)
                                
                                # Append an observation so the LLM knows it succeeded
                                accumulated += f"\n\n[System: Plot saved to {plot_path_str} | Dataset: {dataset_file}]"
                            else:
                                st.error(f"Cannot generate plot: Columns missing or file missing.")
                        except Exception as e:
                            st.error(f"Plot generation failed: {e}")
                            
            if accumulated.strip():
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": accumulated,
                })
                project_manager.save_chat_history(pid, st.session_state.messages)
            
            st.rerun()

        elif status == "rejected" and not msg.get("m2m_handled"):
            msg["m2m_handled"] = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "The M2M data request was rejected.",
            })
            pid = st.session_state.current_project or "default"
            project_manager.save_chat_history(pid, st.session_state.messages)
            st.rerun()

    # Check for pending Cloud Zarr (fast path) approval in this message
    if role == "assistant" and msg.get("zarr_pending"):
        params = msg["zarr_pending"]
        approval_key = msg.get("zarr_approval_key", f"zarr_{i}")
        status = render_m2m_approval(params, approval_key, title="Cloud Zarr Data Request (Fast Path)")

        if status == "accepted" and not msg.get("zarr_handled"):
            msg["zarr_handled"] = True
            # Fast path: open the cloud Zarr store, subset, and process inline.
            with st.spinner("Loading from OOI cloud Zarr store (fast path — no THREDDS wait)..."):
                try:
                    from services.zarr_client import ZarrClient
                    from services.data_processor import DataProcessor

                    pid = st.session_state.current_project or "default"
                    processed_dir = str(project_manager.get_processed_data_dir(pid))

                    ds = ZarrClient().open_dataset(
                        subsite=params["subsite"],
                        node=params["node"],
                        sensor=params["sensor"],
                        method=params["method"],
                        stream=params["stream"],
                        begin_dt=params["begin_dt"],
                        end_dt=params["end_dt"],
                    )
                    res = DataProcessor().process_dataset(ds, processed_dir)
                    cols = ", ".join(res["variables"])
                    netcdf_path = res["netcdf_path"]

                    obs = (
                        f"Observation (Cloud Zarr data imported — fast path, no THREDDS wait):\n"
                        f"Data loaded directly from the OOI cloud Zarr store and processed to NetCDF at: `{netcdf_path}`\n"
                        f"Available variables: {cols}\n"
                        f"You can now generate a plot of this data using the <generate_plot> tool if requested."
                    )
                    st.success(f"Loaded from cloud Zarr store → `{netcdf_path}`")
                except Exception as e:
                    obs = f"Cloud Zarr load failed: {e}. You may retry with an <m2m_request> instead."
                    st.error(f"Zarr load failed: {e}")

            # Continue the agent loop with the observation (auto-plot, etc.)
            llm_mgr = get_llm_manager(st)
            pid = st.session_state.current_project or "default"
            original_msg = st.session_state.messages[0]["content"] if st.session_state.messages else ""

            accumulated = ""
            with st.chat_message("assistant", avatar="🐟"):
                placeholder = st.empty()
                for event in agent_loop_with_m2m_continuation(
                    message=original_msg,
                    project_id=pid,
                    llm_manager=llm_mgr,
                    m2m_observation=obs,
                    prior_accumulated=content,
                    session_messages=st.session_state.messages,
                ):
                    if event["type"] == "token":
                        accumulated += event["text"]
                        display, _, _ = clean_for_display(accumulated)
                        placeholder.markdown(f"<div class='msg-assistant'></div>\n{display}", unsafe_allow_html=True)
                    elif event["type"] == "system":
                        st.info(f"{event['text']}")
                    elif event["type"] == "search_results":
                        with st.expander("🔍 Instrument Search Results", expanded=False):
                            st.code(event["text"], language=None)
                    elif event["type"] == "update_view":
                        vp = event["params"]
                        if "plot" in vp:
                            st.session_state.active_plot = vp["plot"]
                        if "dataset" in vp:
                            st.session_state.active_dataset = vp["dataset"]
                        if "flowchart" in vp:
                            st.session_state.active_flowchart = vp["flowchart"]
                            project_manager.save_flowchart(pid, vp["flowchart"])
                    elif event["type"] == "generate_plot":
                        params = event["params"]
                        dataset_file = params["dataset"]
                        try:
                            from streamlit_utils.plot_renderer import generate_and_save_plot
                            from pathlib import Path
                            plot_dir = Path(dataset_file.split(",")[0].strip()).parent.parent / "plots"
                            plot_dir.mkdir(parents=True, exist_ok=True)
                            import uuid
                            plot_name = f"plot_{uuid.uuid4().hex[:8]}.png"
                            params["force_plot_path"] = str(plot_dir / plot_name)
                            plot_path_str = generate_and_save_plot(params)
                            if plot_path_str:
                                st.session_state.active_plot = plot_path_str
                                st.image(plot_path_str)
                                accumulated += f"\n\n[System: Plot saved to {plot_path_str} | Dataset: {dataset_file}]"
                            else:
                                st.error("Cannot generate plot: Columns missing or file missing.")
                        except Exception as e:
                            st.error(f"Plot generation failed: {e}")

            if accumulated.strip():
                st.session_state.messages.append({"role": "assistant", "content": accumulated})
                project_manager.save_chat_history(pid, st.session_state.messages)
            st.rerun()

        elif status == "rejected" and not msg.get("zarr_handled"):
            msg["zarr_handled"] = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "The cloud Zarr data request was rejected.",
            })
            pid = st.session_state.current_project or "default"
            project_manager.save_chat_history(pid, st.session_state.messages)
            st.rerun()

    # Check for THREDDS import
    if role == "assistant" and msg.get("type") == "thredds_import":
        if not msg.get("imported"):
            st.info(f"Data is ready to be imported from THREDDS:\n`{msg.get('thredds_url')}`")
            if st.button("Import Data from THREDDS", key=f"import_{i}"):
                with st.spinner("Polling THREDDS and downloading data (this may take a few minutes)..."):
                    try:
                        from services.m2m_client import M2MClient
                        from services.data_processor import DataProcessor
                        from core.config import settings
                        from state.project_manager import project_manager

                        client = M2MClient(settings.OOI_USERNAME, settings.OOI_TOKEN)
                        nc_urls = client.check_thredds_status(msg["thredds_url"], max_wait=600)
                        local_files = client.download_netcdf_files(nc_urls, msg["raw_dir"])
                        
                        processor = DataProcessor()
                        res = processor.process_pipeline(local_files, msg["processed_dir"])
                        
                        msg["imported"] = True
                        pid = st.session_state.current_project or "default"
                        project_manager.save_chat_history(pid, st.session_state.messages)
                        
                        cols = ", ".join(res["variables"])
                        netcdf_path = res["netcdf_path"]
                        
                        obs_msg = (
                            f"Observation (Data Imported):\n"
                            f"Data successfully imported and processed to NetCDF at: `{netcdf_path}`\n"
                            f"Available variables: {cols}\n"
                            f"You can now generate a plot of this data using the <generate_plot> tool if requested."
                        )
                        st.session_state.pending_prompt = obs_msg
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to import data: {e}")
        else:
            st.success(f"Data successfully imported from THREDDS.")


# ──────────────────────────────────────────────────────────────
# Chat input
# ──────────────────────────────────────────────────────────────

prompt = st.chat_input("Message OOI RCA Copilot...", key="chat_input")

if prompt and not st.session_state.current_project:
    llm_mgr = get_llm_manager(st)
    with st.spinner("Creating chat..."):
        try:
            title_prompt = (
                f"Generate a 2-4 word chat title for the following request. "
                f"Do not include quotes, prefixes, or any extra text, just the title itself.\n\n"
                f"Request: {prompt}"
            )
            title = llm_mgr.generate_response(title_prompt, is_raw=False)
            title = re.sub(r'<[^>]*>.*?(?:<[^>]*>|\Z)', '', title, flags=re.DOTALL)
            title = re.sub(r'[^\w\s\-]', '', title).strip()
            title = ' '.join(title.split()[:5])
            if not title:
                title = "New Chat"
        except Exception:
            title = "New Chat"

        try:
            purpose_prompt = (
                f"In one short sentence, describe the core objective or context of the following request.\n\n"
                f"Request: {prompt}"
            )
            purpose = llm_mgr.generate_response(purpose_prompt, is_raw=False)
            purpose = re.sub(r'<[^>]*>.*?(?:<[^>]*>|\Z)', '', purpose, flags=re.DOTALL).strip()
        except Exception:
            purpose = "Explore OOI Data"

    project_id = project_manager.create_project(st.session_state.user_id, title, purpose)
    st.session_state.current_project = project_id
    st.session_state.pending_prompt = prompt
    st.rerun()

if getattr(st.session_state, "pending_prompt", None):
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    # ── Add user message ──
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🦀"):
        st.markdown(f"<div class='msg-user'></div>\n{prompt}", unsafe_allow_html=True)

    # ── Generate response ──
    pid = st.session_state.current_project or "default"
    project_manager.current_project_id = pid
    llm_mgr = get_llm_manager(st)

    accumulated = ""
    m2m_request_params = None
    m2m_accumulated = ""
    zarr_request_params = None

    with st.chat_message("assistant", avatar="🐟"):
        placeholder = st.empty()

        for event in agent_loop(
            message=prompt, 
            project_id=pid, 
            llm_manager=llm_mgr, 
            session_messages=st.session_state.messages
        ):
            if event["type"] == "token":
                accumulated += event["text"]
                display, _, _ = clean_for_display(accumulated)
                placeholder.markdown(f"<div class='msg-assistant'></div>\n{display}", unsafe_allow_html=True)

            elif event["type"] == "system":
                st.info(f"{event['text']}")

            elif event["type"] == "search_results":
                with st.expander("🔍 Instrument Search Results", expanded=False):
                    st.code(event["text"], language=None)

            elif event["type"] == "m2m_request":
                m2m_request_params = event["params"]
                m2m_accumulated = event.get("accumulated", accumulated)
                st.info("M2M data request detected — awaiting your approval below.")

            elif event["type"] == "zarr_request":
                zarr_request_params = event["params"]
                st.info("Cloud Zarr (fast path) request detected — awaiting your approval below.")

            elif event["type"] == "generate_plot":
                params = event["params"]
                dataset_file = params["dataset"]
                x_col = params["x_col"]
                y_col = params["y_col"]
                title = params.get("title", f"{y_col} vs {x_col}")
                
                color_col = params.get("color_col")
                static_color = params.get("color", "tab:blue")
                plot_type = params.get("plot_type", "line").lower()
                invert_y = params.get("invert_y", "false").lower() == "true"
                try:
                    from streamlit_utils.plot_renderer import generate_and_save_plot
                    plot_path_str = generate_and_save_plot(params)
                    
                    if plot_path_str:
                        from pathlib import Path
                        plot_path = Path(plot_path_str)
                        plot_name = plot_path.name
                        
                        st.session_state.active_plot = str(plot_path)
                        st.image(str(plot_path))
                        
                        @st.dialog("🖼️ Plot Gallery")
                        def plot_gallery_dialog(pid: str):
                            plot_folder = Path(f"projects/{pid}/plots")
                            if plot_folder.exists():
                                plots = list(plot_folder.glob("*.png"))
                                if plots:
                                    for p in sorted(plots, key=lambda x: x.stat().st_mtime, reverse=True):
                                        st.image(str(p))
                                        with open(p, "rb") as f:
                                            st.download_button("Download", data=f, file_name=p.name, mime="image/png", key=f"dl_{p.name}")
                                        st.markdown("---")
                                else:
                                    st.info("No plots found in this chat yet.")
                            else:
                                st.info("No plots found in this chat yet.")

                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            with open(plot_path, "rb") as f:
                                st.download_button("💾 Download Plot", data=f, file_name=plot_name, mime="image/png", key=f"dl_main_{plot_name}")
                        with col2:
                            if st.button("🖼️ View Gallery", key=f"gal_btn_{plot_name}"):
                                plot_gallery_dialog(pid)
                        with col3:
                            # Dataset Download button
                            ds_list = [c.strip() for c in dataset_file.split(",")]
                            for idx, ds_path_str in enumerate(ds_list):
                                if Path(ds_path_str).exists():
                                    with open(ds_path_str, "rb") as f:
                                        label = "📥 Download Dataset" if len(ds_list) == 1 else f"📥 Download Dataset {idx+1}"
                                        is_nc = ds_path_str.endswith('.nc')
                                        st.download_button(label, data=f, file_name=Path(ds_path_str).name, mime="application/x-netcdf" if is_nc else "text/csv", key=f"dl_ds_{plot_name}_{idx}")
                                
                        accumulated += f"\n\n[System: Plot saved to {plot_path} | Dataset: {dataset_file}]"
                    else:
                        st.error(f"Cannot generate plot: Columns '{x_col}' or '{y_col}' not found or file is missing.")
                except Exception as e:
                    st.error(f"Plot generation failed: {e}")

            elif event["type"] == "render_map":
                params = event["params"]
                lat = params.get("lat")
                lon = params.get("lon")
                title = params.get("title", "Map")
                try:
                    f_lat, f_lon = float(lat), float(lon)
                    bbox = f"{f_lon-0.5}%2C{f_lat-0.5}%2C{f_lon+0.5}%2C{f_lat+0.5}"
                    osm_html = f'''
                    <div style="margin-top: 1rem; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.2);">
                        <div style="background-color: rgba(255,255,255,0.1); padding: 0.5rem 1rem; font-weight: 500;">📍 {title}</div>
                        <iframe width="100%" height="350" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" 
                        src="https://www.openstreetmap.org/export/embed.html?bbox={bbox}&amp;layer=mapnik&amp;marker={f_lat}%2C{f_lon}">
                        </iframe>
                    </div>
                    '''
                    st.markdown(osm_html, unsafe_allow_html=True)
                    accumulated += f"\n\n[System: Map rendered at lat={lat}, lon={lon}, title={title}]"
                except Exception as e:
                    st.error(f"Map generation failed: {e}")

            elif event["type"] == "update_view":
                view_params = event["params"]
                if "plot" in view_params:
                    st.session_state.active_plot = view_params["plot"]
                if "dataset" in view_params:
                    st.session_state.active_dataset = view_params["dataset"]
                if "flowchart" in view_params:
                    st.session_state.active_flowchart = view_params["flowchart"]
                    project_manager.save_flowchart(pid, view_params["flowchart"])

    # ── Save the assistant message ──
    msg_data = {"role": "assistant", "content": accumulated}

    if m2m_request_params:
        approval_key = str(uuid.uuid4())[:8]
        msg_data["m2m_pending"] = m2m_request_params
        msg_data["m2m_approval_key"] = approval_key
        msg_data["m2m_handled"] = False

    if zarr_request_params:
        approval_key = str(uuid.uuid4())[:8]
        msg_data["zarr_pending"] = zarr_request_params
        msg_data["zarr_approval_key"] = approval_key
        msg_data["zarr_handled"] = False

    st.session_state.messages.append(msg_data)
    project_manager.save_chat_history(pid, st.session_state.messages)

    if m2m_request_params or zarr_request_params:
        st.rerun()  # Rerun to render the approval widget
