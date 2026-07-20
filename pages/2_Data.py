"""
Data — View processed data, plots, and pipeline flowcharts.

Displays the active project's data outputs across three tabs:
Pipeline (Mermaid flowchart), Plot (matplotlib images), and Data (CSV table).
"""

import sys
from pathlib import Path

# Ensure backend is importable
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import streamlit as st

st.set_page_config(
    page_title="Data — OOI RCA Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_utils.session import init_session_state
from streamlit_utils.ui_components import (
    inject_custom_css,
    render_project_sidebar,
    render_flowchart_mermaid,
)

init_session_state(st)
inject_custom_css()

# ── Sidebar ──
render_project_sidebar()

from state.project_manager import project_manager

# ── Header ──
st.title("Data Viewer")

pid = st.session_state.current_project
if not pid:
    st.info("Select or create a project to view its data.")
    st.stop()

st.markdown(f"**Project:** `{pid}`")
st.markdown("---")

# ── Tabs ──
tab_pipeline, tab_plot, tab_data, tab_files = st.tabs([
    "Pipeline", "Plots", "Data (CSV)", "Files"
])


# ── Pipeline Tab ──
with tab_pipeline:
    flowchart_steps = st.session_state.get("active_flowchart", "")
    if not flowchart_steps:
        flowchart_steps = project_manager.load_flowchart(pid)
        st.session_state.active_flowchart = flowchart_steps

    render_flowchart_mermaid(flowchart_steps)

    if flowchart_steps:
        st.caption(f"Steps: {flowchart_steps}")


# ── Plot Tab ──
with tab_plot:
    plot_path = st.session_state.get("active_plot")
    plots_dir = project_manager.get_plots_dir(pid)

    if plot_path:
        full_path = Path(plot_path)
        if not full_path.is_absolute():
            full_path = Path(__file__).resolve().parent.parent / plot_path

        if full_path.exists():
            st.image(str(full_path), width='stretch')
            st.caption(f"{full_path.name}")
        else:
            st.warning(f"Plot file not found: `{plot_path}`")

    # Also list all plots in the project's plots directory
    if plots_dir.exists():
        plot_files = sorted(plots_dir.glob("*.png")) + sorted(plots_dir.glob("*.jpg"))
        if plot_files:
            st.markdown("##### All Plots")
            selected_plot = st.selectbox(
                "Select a plot",
                [f.name for f in plot_files],
                key="plot_selector",
            )
            if selected_plot:
                selected_path = plots_dir / selected_plot
                st.image(str(selected_path), width='stretch')
        elif not plot_path:
            st.info("No plots generated yet. Use the Chat page to generate visualizations.")
    elif not plot_path:
        st.info("No plots generated yet. Use the Chat page to generate visualizations.")


# ── Data Table Tab ──
with tab_data:
    dataset_path = st.session_state.get("active_dataset")
    processed_dir = project_manager.get_processed_data_dir(pid)

    dataset_files = []
    if processed_dir.exists():
        dataset_files = sorted(processed_dir.glob("*.nc"))
        if not dataset_files: # Fallback for old projects
            dataset_files = sorted(processed_dir.glob("*.csv"))

    if dataset_path:
        full_ds_path = Path(dataset_path)
        if not full_ds_path.is_absolute():
            full_ds_path = Path(__file__).resolve().parent.parent / dataset_path

        if full_ds_path.exists():
            from streamlit_utils.data_utils import load_netcdf, load_csv
            try:
                is_nc = full_ds_path.suffix == '.nc'
                
                st.markdown(f"##### {full_ds_path.name}")
                
                if is_nc:
                    ds = load_netcdf(full_ds_path)
                    
                    # Display Metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Variables", f"{len(ds.data_vars):,}")
                    with col2:
                        dims_str = ", ".join([f"{k}: {v}" for k, v in ds.sizes.items()])
                        st.metric("Dimensions", dims_str)
                    with col3:
                        st.metric("File Size", f"{full_ds_path.stat().st_size / 1e6:.2f} MB")
                        
                    with st.expander("Attributes & Metadata", expanded=False):
                        st.json(ds.attrs)

                    # Preview table (take first 10,000 for preview to save memory)
                    # Convert only a slice to avoid loading entire dataset into memory
                    # We just take a subset using isel on the primary dimension (usually time)
                    primary_dim = list(ds.dims)[0] if ds.dims else None
                    if primary_dim and ds.sizes[primary_dim] > 10000:
                        st.warning(f"⚠️ Dataset is very large. Only showing the first 10,000 '{primary_dim}' rows in preview.")
                        preview_df = ds.isel({primary_dim: slice(0, 10000)}).to_dataframe().reset_index()
                    else:
                        preview_df = ds.to_dataframe().reset_index()
                        
                    st.dataframe(preview_df, width='stretch', height=400)
                    
                    # CSV Download Button
                    # To avoid memory crashes, we write the full dataframe to a temporary file, then serve it
                    if st.button("Generate & Download CSV"):
                        with st.spinner("Converting NetCDF to CSV for download..."):
                            import tempfile
                            import os
                            fd, temp_csv = tempfile.mkstemp(suffix=".csv")
                            os.close(fd)
                            
                            # Convert to DataFrame (this may spike memory briefly if very large)
                            # But xarray will handle it better than pandas
                            full_df = ds.to_dataframe().reset_index()
                            full_df.to_csv(temp_csv, index=False)
                            
                            with open(temp_csv, "rb") as f:
                                st.download_button(
                                    label="Download CSV Ready",
                                    data=f,
                                    file_name=full_ds_path.with_suffix('.csv').name,
                                    mime="text/csv",
                                )
                else:
                    df = load_csv(full_ds_path)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", f"{len(df.columns):,}")
                    with col3:
                        st.metric("Size", f"{full_ds_path.stat().st_size / 1e6:.2f} MB")

                    if len(df) > 10000:
                        st.warning(f"⚠️ Dataset is very large ({len(df):,} rows). Only showing the first 10,000 rows.")
                        st.dataframe(df.head(10000), width='stretch', height=500)
                    else:
                        st.dataframe(df, width='stretch', height=500)

                    with open(full_ds_path, "rb") as f:
                        st.download_button("Download CSV", f, file_name=full_ds_path.name, mime="text/csv")

            except Exception as e:
                st.error(f"Error loading Dataset: {e}")
        else:
            st.warning(f"Dataset file not found: `{dataset_path}`")

    if dataset_files:
        st.markdown("##### Processed Datasets")
        selected_ds = st.selectbox(
            "Select a Dataset file",
            [f.name for f in dataset_files],
            key="ds_selector",
        )
        if selected_ds:
            from streamlit_utils.data_utils import load_netcdf, load_csv
            selected_full = processed_dir / selected_ds
            try:
                if selected_full.suffix == '.nc':
                    ds = load_netcdf(selected_full)
                    primary_dim = list(ds.dims)[0] if ds.dims else None
                    if primary_dim and ds.sizes[primary_dim] > 10000:
                        st.warning(f"⚠️ Dataset is very large. Only showing the first 10,000 '{primary_dim}' rows in preview.")
                        preview_df = ds.isel({primary_dim: slice(0, 10000)}).to_dataframe().reset_index()
                    else:
                        preview_df = ds.to_dataframe().reset_index()
                    st.dataframe(preview_df, width='stretch', height=400)
                    
                    with open(selected_full, "rb") as f:
                        st.download_button(
                            "Download NetCDF",
                            data=f,
                            file_name=selected_full.name,
                            mime="application/x-netcdf",
                            key=f"dl_sel_nc_{selected_full.name}"
                        )
                else:
                    df = load_csv(selected_full)
                    if len(df) > 10000:
                        st.warning(f"⚠️ Dataset is very large ({len(df):,} rows). Only showing the first 10,000 rows.")
                        st.dataframe(df.head(10000), width='stretch', height=400)
                    else:
                        st.dataframe(df, width='stretch', height=400)
                    
                    with open(selected_full, "rb") as f:
                        st.download_button("Download Selected CSV", data=f, file_name=selected_full.name, mime="text/csv", key=f"dl_sel_{selected_full.name}")
                    
                # QARTOD Summary Display
                import json
                qc_path = selected_full.parent / "qc_summary.json"
                if qc_path.exists():
                    try:
                        qc_data = json.loads(qc_path.read_text())
                        if qc_data:
                            with st.expander("🛡️ Data Quality Control (QARTOD) Summary", expanded=False):
                                st.markdown("The following suspect or failing data points were automatically filtered out during processing based on OOI QARTOD flags:")
                                cols = st.columns(min(len(qc_data), 4) or 1)
                                for i, (var_name, metrics) in enumerate(qc_data.items()):
                                    with cols[i % len(cols)]:
                                        st.metric(
                                            label=var_name, 
                                            value=f"-{metrics['dropped']} pts",
                                            delta=f"{metrics['percent_dropped']}% dropped",
                                            delta_color="inverse"
                                        )
                    except Exception as e:
                        pass
            except Exception as e:
                st.error(f"Error loading {selected_ds}: {e}")
    elif not dataset_path:
        st.info("No data files yet. Use the Chat page to process data.")


# ── Files Tab ──
with tab_files:
    st.markdown("##### Raw Data")
    raw_dir = project_manager.get_raw_data_dir(pid)
    if raw_dir.exists():
        raw_files = sorted(raw_dir.iterdir())
        if raw_files:
            for f in raw_files:
                if f.is_file():
                    size_mb = f.stat().st_size / 1e6
                    st.text(f"  📄 {f.name}  ({size_mb:.2f} MB)")
        else:
            st.caption("No raw data files.")
    else:
        st.caption("No raw data directory.")

    st.markdown("##### Processed Data")
    if processed_dir.exists():
        proc_files = sorted(processed_dir.iterdir())
        if proc_files:
            for f in proc_files:
                if f.is_file():
                    size_mb = f.stat().st_size / 1e6
                    st.text(f"  📄 {f.name}  ({size_mb:.2f} MB)")
        else:
            st.caption("No processed data files.")
    else:
        st.caption("No processed data directory.")

    st.markdown("##### Scripts")
    scripts_dir = project_manager.get_scripts_dir(pid)
    if scripts_dir.exists():
        script_files = sorted(scripts_dir.glob("*.py"))
        if script_files:
            selected_script = st.selectbox(
                "View script source",
                [f.name for f in script_files],
                key="script_selector",
            )
            if selected_script:
                script_content = (scripts_dir / selected_script).read_text(encoding="utf-8")
                st.code(script_content, language="python")
        else:
            st.caption("No scripts generated yet.")
    else:
        st.caption("No scripts directory.")
