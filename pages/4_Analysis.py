"""
Advanced Analysis Workspace — OOI RCA Copilot

A full-featured interactive data analysis environment. Users can:
  - Load project datasets or upload their own CSV files
  - Select from multiple plot types (line, scatter, histogram, box, heatmap, depth profile)
  - Apply data transformations (smoothing, resampling, outlier removal)
  - Customize titles, colors, axis limits, and styling
  - Export plots as PNG
  - Send findings to the AI Copilot for hypothesis generation
"""

import sys
import io
import uuid
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import streamlit as st

st.set_page_config(
    page_title="Analysis — OOI RCA Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

from streamlit_utils.session import init_session_state
from streamlit_utils.ui_components import (
    inject_custom_css,
    render_project_sidebar,
)
from state.project_manager import project_manager

init_session_state(st)
inject_custom_css()

# ── Sidebar ──
render_project_sidebar()

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

PLOT_TYPES = [
    "Line",
    "Scatter",
    "Histogram",
    "Box Plot",
    "Depth Profile (inverted Y)",
    "Heatmap (pivot)",
    "Multi-Y Overlay",
]

COLORMAPS = ["viridis", "plasma", "inferno", "cividis", "coolwarm", "turbo", "ocean", "deep"]

RESAMPLE_OPTIONS = {
    "1 Minute": "1min",
    "5 Minutes": "5min",
    "15 Minutes": "15min",
    "Hourly": "1h",
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS",
}

# Known science-relevant columns (auto-detect for smart defaults)
SCIENCE_COLS = [
    "sea_water_temperature", "sea_water_practical_salinity", "sea_water_density",
    "sea_water_pressure", "sea_water_electrical_conductivity", "conductivity_millisiemens",
    "chlorophyll_a", "optical_backscatter", "dissolved_oxygen", "seawater_ph",
    "pco2_seawater", "nitrate_concentration", "turbidity", "par",
    "water_velocity_eastward", "water_velocity_northward", "water_velocity_upward",
    "bottom_pressure", "tilt_x", "tilt_y",
]


# ══════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════

from streamlit_utils.data_utils import load_csv, load_netcdf

def get_numeric_cols(ds_or_df) -> list[str]:
    if isinstance(ds_or_df, pd.DataFrame):
        return [c for c in ds_or_df.columns if pd.api.types.is_numeric_dtype(ds_or_df[c])]
    else: # xarray Dataset
        return [str(v) for v in ds_or_df.variables if ds_or_df[v].dtype.kind in 'iuf']

def get_datetime_cols(ds_or_df) -> list[str]:
    if isinstance(ds_or_df, pd.DataFrame):
        return [c for c in ds_or_df.columns if pd.api.types.is_datetime64_any_dtype(ds_or_df[c])]
    else: # xarray Dataset
        return [str(v) for v in ds_or_df.variables if ds_or_df[v].dtype.kind == 'M']

def smart_default_y(ds_or_df) -> int:
    """Return the index of the first science-relevant numeric column, or 0."""
    numeric = get_numeric_cols(ds_or_df)
    for sci in SCIENCE_COLS:
        if sci in numeric:
            return numeric.index(sci)
    return 0


# ══════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════

st.title("🔬 Advanced Analysis Workspace")

# ══════════════════════════════════════════════════════════════
# Step 1 — Data Source
# ══════════════════════════════════════════════════════════════

st.markdown("### 📁 Data Source")

source_tab1, source_tab2 = st.tabs(["Project Dataset", "Upload Your Own NetCDF/CSV"])

ds = None
df_static = None # For CSV fallbacks
data_label = ""
is_nc = False

with source_tab1:
    pid = st.session_state.current_project
    if not pid:
        st.info("Select a project in the sidebar first, or upload your own file in the next tab.")
    else:
        project_dir = project_manager.base_dir / pid
        data_dir = project_dir / "processed_data"
        if data_dir.exists():
            files = sorted(data_dir.glob("*.nc"))
            if not files:
                files = sorted(data_dir.glob("*.csv")) # Fallback
            
            if files:
                names = [f.name for f in files]
                chosen = st.selectbox("Dataset", names, key="proj_dataset")
                file_path = data_dir / chosen
                if chosen.endswith(".nc"):
                    ds = load_netcdf(file_path)
                    is_nc = True
                else:
                    df_static = load_csv(file_path)
                data_label = f"{pid} / {chosen}"
            else:
                st.warning("No datasets found in this project's processed_data folder.")
        else:
            st.warning("This project has no imported data yet. Use the Chat to pull data first.")

with source_tab2:
    uploaded = st.file_uploader("Drop a file here (.nc or .csv)", type=["nc", "csv"], key="ds_upload")
    if uploaded is not None:
        if uploaded.name.endswith(".nc"):
            ds = load_netcdf(uploaded.getvalue())
            is_nc = True
        else:
            df_static = load_csv(uploaded.getvalue(), uploaded.name)
        data_label = f"Upload: {uploaded.name}"

if ds is None and df_static is None:
    st.stop()

# ── Quick data summary ──
data_obj = ds if is_nc else df_static
with st.expander(f"📋 Data Summary — {data_label}", expanded=False):
    c1, c2, c3 = st.columns(3)
    if is_nc:
        primary_dim = list(ds.dims)[0] if ds.dims else None
        num_rows = ds.sizes[primary_dim] if primary_dim else 0
        c1.metric("Records", f"{num_rows:,}")
        c2.metric("Variables", f"{len(ds.data_vars)}")
    else:
        c1.metric("Rows", f"{len(df_static):,}")
        c2.metric("Columns", f"{len(df_static.columns)}")
        
    dt_cols = get_datetime_cols(data_obj)
    if dt_cols:
        if is_nc:
            tmin = pd.to_datetime(ds[dt_cols[0]].min().values)
            tmax = pd.to_datetime(ds[dt_cols[0]].max().values)
        else:
            tmin = df_static[dt_cols[0]].min()
            tmax = df_static[dt_cols[0]].max()
        c3.metric("Time Span", f"{(tmax - tmin).days} days")

    if is_nc:
        primary_dim = list(ds.dims)[0] if ds.dims else None
        if primary_dim and ds.sizes[primary_dim] > 10000:
            preview = ds.isel({primary_dim: slice(0, 100)}).to_dataframe().reset_index()
        else:
            preview = ds.to_dataframe().reset_index().head(100)
    else:
        preview = df_static.head(100)
    st.dataframe(preview, width='stretch', height=200)


# ══════════════════════════════════════════════════════════════
# Step 2 — Plot Type & Variables
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 📊 Plot Configuration")

numeric_cols = get_numeric_cols(data_obj)
datetime_cols = get_datetime_cols(data_obj)
if is_nc:
    all_cols = list(ds.data_vars) + list(ds.coords)
else:
    all_cols = list(df_static.columns)

# ── Pre-baked Recipes ──
# Each recipe: (label, plot_type, x_col, y_col, color_col, title, invert_y)
RECIPES = [
    ("— Custom (manual selection) —", None, None, None, None, None, False),
    ("🌡️ Temperature vs Time",         "Line",    "time", "sea_water_temperature", None, "Sea Water Temperature Time Series", False),
    ("🧂 Salinity vs Time",            "Line",    "time", "sea_water_practical_salinity", None, "Practical Salinity Time Series", False),
    ("📊 T-S Diagram (Scatter)",       "Scatter", "sea_water_practical_salinity", "sea_water_temperature", "sea_water_density", "T-S Diagram", False),
    ("📏 Temperature Depth Profile",   "Depth Profile (inverted Y)", "sea_water_pressure", "sea_water_temperature", None, "Temperature Depth Profile", True),
    ("📏 Salinity Depth Profile",      "Depth Profile (inverted Y)", "sea_water_pressure", "sea_water_practical_salinity", None, "Salinity Depth Profile", True),
    ("📏 Density Depth Profile",       "Depth Profile (inverted Y)", "sea_water_pressure", "sea_water_density", None, "Density Depth Profile", True),
    ("⚡ Conductivity vs Time",        "Line",    "time", "sea_water_electrical_conductivity", None, "Electrical Conductivity Time Series", False),
    ("📏 Conductivity Depth Profile",  "Depth Profile (inverted Y)", "sea_water_pressure", "sea_water_electrical_conductivity", None, "Conductivity Depth Profile", True),
    ("🌊 Density vs Time",             "Line",    "time", "sea_water_density", None, "Seawater Density Time Series", False),
    ("🌊 Density vs Salinity",         "Scatter", "sea_water_practical_salinity", "sea_water_density", "sea_water_temperature", "Density vs Salinity (colored by Temp)", False),
    ("🟢 Chlorophyll-a vs Time",       "Line",    "time", "chlorophyll_a", None, "Chlorophyll-a Concentration", False),
    ("💨 Dissolved Oxygen vs Time",    "Line",    "time", "dissolved_oxygen", None, "Dissolved Oxygen Time Series", False),
    ("📏 Dissolved Oxygen Profile",    "Depth Profile (inverted Y)", "sea_water_pressure", "dissolved_oxygen", None, "DO₂ Depth Profile", True),
    ("🔬 pH vs Time",                  "Line",    "time", "seawater_ph", None, "Seawater pH Time Series", False),
    ("🫧 pCO₂ vs Time",               "Line",    "time", "pco2_seawater", None, "pCO₂ Time Series", False),
    ("🧪 Nitrate vs Time",            "Line",    "time", "nitrate_concentration", None, "Nitrate Concentration", False),
    ("📏 Nitrate Depth Profile",       "Depth Profile (inverted Y)", "sea_water_pressure", "nitrate_concentration", None, "Nitrate Depth Profile", True),
    ("💡 PAR vs Time",                 "Line",    "time", "par", None, "Photosynthetically Active Radiation", False),
    ("🌫️ Turbidity vs Time",          "Line",    "time", "turbidity", None, "Turbidity Time Series", False),
    ("🌫️ Optical Backscatter vs Time","Line",    "time", "optical_backscatter", None, "Optical Backscatter", False),
    ("🌊 Pressure vs Time",           "Line",    "time", "sea_water_pressure", None, "Pressure (Depth Proxy) Time Series", False),
    ("🔀 T-S Diagram (Line)",         "Line",    "sea_water_practical_salinity", "sea_water_temperature", None, "T-S Diagram (Line)", False),
    ("📊 Temp vs Density (Scatter)",   "Scatter", "sea_water_temperature", "sea_water_density", "sea_water_practical_salinity", "Temp vs Density (colored by Salinity)", False),
    ("🌊 Pressure vs Density",        "Scatter", "sea_water_density", "sea_water_pressure", "sea_water_temperature", "Pressure vs Density", False),
    ("📈 Temp + Salinity Overlay",     "Multi-Y Overlay", "time", None, None, "Temperature & Salinity Co-plot", False),
]

# Filter recipes to only show ones whose required columns exist in the data
def recipe_available(r):
    if r[1] is None:  # Custom
        return True
    x_ok = r[2] is None or r[2] in all_cols
    y_ok = r[3] is None or r[3] in numeric_cols
    return x_ok and y_ok

available_recipes = [r for r in RECIPES if recipe_available(r)]
recipe_labels = [r[0] for r in available_recipes]

selected_recipe_label = st.selectbox("🍳 Quick Recipe (or choose Custom)", recipe_labels, key="recipe_sel")
selected_recipe = next(r for r in available_recipes if r[0] == selected_recipe_label)
is_custom = selected_recipe[1] is None

# Detect recipe change and clear downstream widget keys so new defaults apply
_prev_recipe = st.session_state.get("_prev_recipe")
if _prev_recipe != selected_recipe_label:
    st.session_state["_prev_recipe"] = selected_recipe_label
    for k in ["plot_type", "xy_x", "xy_y", "xy_color", "multi_x", "multi_y",
              "hist_col", "box_cols", "hm_x", "hm_y", "hm_val", "custom_title"]:
        st.session_state.pop(k, None)
    st.rerun()

# ── Two-column layout: variables on left, styling on right ──
left, right = st.columns([3, 2])

with left:
    # Plot type — from recipe or manual
    if is_custom:
        plot_type = st.selectbox("Plot Type", PLOT_TYPES, key="plot_type")
    else:
        recipe_plot_type = selected_recipe[1]
        # Allow user to override the render style
        plot_type = st.selectbox(
            "Plot Type",
            PLOT_TYPES,
            index=PLOT_TYPES.index(recipe_plot_type) if recipe_plot_type in PLOT_TYPES else 0,
            key="plot_type",
        )

    if plot_type == "Histogram":
        hist_col = st.selectbox("Variable", numeric_cols, key="hist_col")
        hist_bins = st.slider("Bins", 10, 200, 50, key="hist_bins")
    elif plot_type == "Box Plot":
        box_cols = st.multiselect("Variables to compare", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols, key="box_cols")
    elif plot_type == "Heatmap (pivot)":
        hm_x = st.selectbox("X (categorical/time)", all_cols, key="hm_x")
        hm_y = st.selectbox("Y (categorical)", all_cols, key="hm_y")
        hm_val = st.selectbox("Value (numeric)", numeric_cols, key="hm_val")
    elif plot_type == "Multi-Y Overlay":
        default_x_opts = datetime_cols + numeric_cols if datetime_cols else numeric_cols
        if not is_custom and selected_recipe[2] and selected_recipe[2] in default_x_opts:
            multi_x_idx = default_x_opts.index(selected_recipe[2])
        else:
            multi_x_idx = 0
        x_axis = st.selectbox("X-Axis", default_x_opts, index=multi_x_idx, key="multi_x")

        # For recipes like "Temp + Salinity Overlay", provide smart defaults
        default_multi_y = []
        if not is_custom and selected_recipe[0] == "📈 Temp + Salinity Overlay":
            for c in ["sea_water_temperature", "sea_water_practical_salinity"]:
                if c in numeric_cols:
                    default_multi_y.append(c)
        if not default_multi_y:
            default_multi_y = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        y_axes = st.multiselect("Y-Axes (select 2–4)", numeric_cols, default=default_multi_y, key="multi_y")
    else:
        # Line, Scatter, Depth Profile
        default_x_options = datetime_cols + numeric_cols if datetime_cols else all_cols

        # Apply recipe defaults for x/y/color
        if not is_custom and selected_recipe[2] and selected_recipe[2] in default_x_options:
            default_x_idx = default_x_options.index(selected_recipe[2])
        else:
            default_x_idx = 0

        if not is_custom and selected_recipe[3] and selected_recipe[3] in numeric_cols:
            default_y_idx = numeric_cols.index(selected_recipe[3])
        else:
            default_y_idx = smart_default_y(data_obj)

        x_axis = st.selectbox("X-Axis", default_x_options, index=default_x_idx, key="xy_x")
        y_axis = st.selectbox("Y-Axis", numeric_cols, index=default_y_idx, key="xy_y")

        color_options = ["None"] + numeric_cols + [c for c in all_cols if c not in numeric_cols]
        if not is_custom and selected_recipe[4] and selected_recipe[4] in color_options:
            default_color_idx = color_options.index(selected_recipe[4])
        else:
            default_color_idx = 0
        color_col = st.selectbox("Color by (optional)", color_options, index=default_color_idx, key="xy_color")

with right:
    st.markdown("**Styling**")
    recipe_title = selected_recipe[5] if not is_custom and selected_recipe[5] else ""
    custom_title = st.text_input("Title (leave blank for auto)", value=recipe_title, key="custom_title")
    user_color = st.color_picker("Primary Color", "#1f77b4", key="primary_color")
    cmap_choice = st.selectbox("Colormap", COLORMAPS, key="cmap_choice")
    
    st.markdown("**Axis Limits** (leave blank for auto)")
    lc1, lc2 = st.columns(2)
    with lc1:
        x_min_str = st.text_input("X Min", key="xmin")
        y_min_str = st.text_input("Y Min", key="ymin")
    with lc2:
        x_max_str = st.text_input("X Max", key="xmax")
        y_max_str = st.text_input("Y Max", key="ymax")


# ══════════════════════════════════════════════════════════════
# Step 3 — Transformations
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🔧 Transformations")

tc1, tc2, tc3 = st.columns(3)

with tc1:
    do_smooth = st.checkbox("Rolling Average", key="do_smooth")
    if do_smooth:
        smooth_window = st.number_input("Window (points)", 2, 5000, 10, key="smooth_win")

with tc2:
    do_resample = st.checkbox("Resample (requires time column)", key="do_resample")
    if do_resample:
        resample_label = st.selectbox("Frequency", list(RESAMPLE_OPTIONS.keys()), key="resample_freq")

with tc3:
    do_outlier = st.checkbox("Remove Outliers (IQR)", key="do_outlier")
    if do_outlier:
        iqr_factor = st.number_input("IQR multiplier", 1.0, 5.0, 1.5, step=0.1, key="iqr_factor")


# ══════════════════════════════════════════════════════════════
# Determine which columns are actually plotted (for targeted transforms)
# ══════════════════════════════════════════════════════════════

_active_y_cols = []
if plot_type in ("Line", "Scatter", "Depth Profile (inverted Y)"):
    _active_y_cols = [y_axis]
elif plot_type == "Multi-Y Overlay":
    _active_y_cols = list(y_axes) if y_axes else []
elif plot_type == "Histogram":
    _active_y_cols = [hist_col]
elif plot_type == "Box Plot":
    _active_y_cols = list(box_cols) if box_cols else []
elif plot_type == "Heatmap (pivot)":
    _active_y_cols = [hm_val]

# ══════════════════════════════════════════════════════════════
# Apply transforms (optimized — only touch plotted columns)
# ══════════════════════════════════════════════════════════════

cols_to_extract = set(_active_y_cols)
if plot_type not in ("Box Plot", "Histogram") and x_axis:
    cols_to_extract.add(x_axis)
if plot_type in ("Scatter", "Depth Profile (inverted Y)") and color_col and color_col != "None":
    cols_to_extract.add(color_col)
if plot_type == "Heatmap (pivot)":
    cols_to_extract.update([hm_x, hm_y, hm_val])
if datetime_cols and datetime_cols[0]:
    cols_to_extract.add(datetime_cols[0])

# Ensure all columns exist in data_obj
if is_nc:
    valid_cols = [c for c in cols_to_extract if c in ds.variables]
    if valid_cols:
        # Check if the dataset is massive. If so, and we aren't resampling, warn and slice
        primary_dim = list(ds.dims)[0] if ds.dims else None
        if primary_dim and ds.sizes[primary_dim] > 500000 and not do_resample:
            st.warning("⚠️ Dataset is extremely large. Slicing to the first 500,000 points to prevent browser crash. Please enable Resampling for full dataset overview.")
            plot_df = ds[valid_cols].isel({primary_dim: slice(0, 500000)}).to_dataframe().reset_index()
        else:
            plot_df = ds[valid_cols].to_dataframe().reset_index()
    else:
        plot_df = pd.DataFrame()
else:
    valid_cols = [c for c in cols_to_extract if c in df_static.columns]
    plot_df = df_static[valid_cols].copy()

# Sort by time if available
if datetime_cols:
    plot_df = plot_df.sort_values(datetime_cols[0])

# Resample
if do_resample and datetime_cols:
    freq = RESAMPLE_OPTIONS[resample_label]
    tcol = datetime_cols[0]
    plot_df.set_index(tcol, inplace=True)
    plot_df = plot_df.resample(freq).mean(numeric_only=True).reset_index()

# Outlier removal — vectorized, only on active columns
if do_outlier and _active_y_cols:
    active_in_df = [c for c in _active_y_cols if c in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[c])]
    if active_in_df:
        subset = plot_df[active_in_df]
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        mask = ((subset >= q1 - iqr_factor * iqr) & (subset <= q3 + iqr_factor * iqr)).all(axis=1)
        plot_df = plot_df[mask]

# Rolling average — only on active columns
smoothed_map = {}
if do_smooth and _active_y_cols:
    for col in _active_y_cols:
        if col in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[col]):
            smoothed_name = f"{col}_smooth"
            plot_df[smoothed_name] = plot_df[col].rolling(window=smooth_window, min_periods=1).mean()
            smoothed_map[col] = smoothed_name

def resolve_col(col_name):
    """Return the smoothed version of a column if smoothing is active."""
    if do_smooth and col_name in smoothed_map:
        return smoothed_map[col_name]
    return col_name


# ══════════════════════════════════════════════════════════════
# Step 4 — Render Plot
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🖼️ Plot Preview")

try:
    fig, ax = plt.subplots(figsize=(12, 5))
    auto_title = ""

    # ── LINE ──
    if plot_type == "Line":
        yc = resolve_col(y_axis)
        ax.plot(plot_df[x_axis], plot_df[yc], color=user_color, linewidth=0.8, alpha=0.9)
        auto_title = f"{y_axis} vs {x_axis}"
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

    # ── SCATTER ──
    elif plot_type == "Scatter":
        yc = resolve_col(y_axis)
        if color_col != "None" and color_col in plot_df.columns:
            sc = ax.scatter(plot_df[x_axis], plot_df[yc], c=plot_df[color_col], cmap=cmap_choice, s=8, alpha=0.7)
            plt.colorbar(sc, ax=ax, label=color_col, pad=0.01)
        else:
            ax.scatter(plot_df[x_axis], plot_df[yc], color=user_color, s=8, alpha=0.7)
        auto_title = f"{y_axis} vs {x_axis}"
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

    # ── DEPTH PROFILE ──
    elif plot_type == "Depth Profile (inverted Y)":
        yc = resolve_col(y_axis)
        if color_col != "None" and color_col in plot_df.columns:
            sc = ax.scatter(plot_df[yc], plot_df[x_axis], c=plot_df[color_col], cmap=cmap_choice, s=8, alpha=0.7)
            plt.colorbar(sc, ax=ax, label=color_col, pad=0.01)
        else:
            ax.scatter(plot_df[yc], plot_df[x_axis], color=user_color, s=8, alpha=0.7)
        ax.invert_yaxis()
        auto_title = f"Depth Profile: {y_axis}"
        ax.set_xlabel(y_axis)
        ax.set_ylabel(x_axis)

    # ── HISTOGRAM ──
    elif plot_type == "Histogram":
        ax.hist(plot_df[hist_col].dropna(), bins=hist_bins, color=user_color, alpha=0.8, edgecolor="black", linewidth=0.3)
        auto_title = f"Distribution of {hist_col}"
        ax.set_xlabel(hist_col)
        ax.set_ylabel("Count")

    # ── BOX PLOT ──
    elif plot_type == "Box Plot":
        if box_cols:
            data_to_box = [plot_df[c].dropna().values for c in box_cols]
            bp = ax.boxplot(data_to_box, labels=box_cols, patch_artist=True)
            colors_cycle = plt.cm.tab10.colors
            for patch, color in zip(bp["boxes"], colors_cycle):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            auto_title = "Variable Comparison (Box Plot)"

    # ── HEATMAP ──
    elif plot_type == "Heatmap (pivot)":
        try:
            pivot = plot_df.pivot_table(index=hm_y, columns=hm_x, values=hm_val, aggfunc="mean")
            im = ax.imshow(pivot.values, cmap=cmap_choice, aspect="auto")
            plt.colorbar(im, ax=ax, label=hm_val, pad=0.01)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(pivot.index, fontsize=7)
            auto_title = f"Heatmap: {hm_val} by {hm_x} × {hm_y}"
        except Exception as e:
            st.warning(f"Could not create heatmap pivot: {e}")

    # ── MULTI-Y OVERLAY ──
    elif plot_type == "Multi-Y Overlay":
        if y_axes:
            colors_cycle = plt.cm.tab10.colors
            lines = []
            labels = []
            for idx_y, yname in enumerate(y_axes):
                yc = resolve_col(yname)
                if idx_y == 0:
                    line, = ax.plot(plot_df[x_axis], plot_df[yc], color=colors_cycle[0], linewidth=0.8, alpha=0.9)
                    ax.set_ylabel(yname, color=colors_cycle[0])
                    ax.tick_params(axis="y", labelcolor=colors_cycle[0])
                else:
                    ax2 = ax.twinx()
                    if idx_y > 1:
                        ax2.spines["right"].set_position(("axes", 1.0 + 0.12 * (idx_y - 1)))
                    line, = ax2.plot(plot_df[x_axis], plot_df[yc], color=colors_cycle[idx_y % len(colors_cycle)], linewidth=0.8, alpha=0.9)
                    ax2.set_ylabel(yname, color=colors_cycle[idx_y % len(colors_cycle)])
                    ax2.tick_params(axis="y", labelcolor=colors_cycle[idx_y % len(colors_cycle)])
                lines.append(line)
                labels.append(yname)
            ax.legend(lines, labels, loc="upper left", fontsize=8)
            ax.set_xlabel(x_axis)
            auto_title = " / ".join(y_axes) + f" vs {x_axis}"

    # ── Common finishing ──
    title_text = custom_title if custom_title else auto_title
    ax.set_title(title_text, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()

    # Apply custom axis limits
    try:
        if x_min_str and x_max_str:
            if datetime_cols and x_axis in [resolve_col(c) for c in datetime_cols]:
                ax.set_xlim(pd.to_datetime(x_min_str), pd.to_datetime(x_max_str))
            else:
                ax.set_xlim(float(x_min_str), float(x_max_str))
    except Exception:
        pass
    try:
        if y_min_str and y_max_str:
            ax.set_ylim(float(y_min_str), float(y_max_str))
    except Exception:
        pass

    st.pyplot(fig)

    # ── Export ──
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "💾 Download Plot (PNG)",
            data=buf,
            file_name=f"analysis_{uuid.uuid4().hex[:6]}.png",
            mime="image/png",
            key="download_plot",
        )
    with exp2:
        csv_buf = plot_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Transformed Data (CSV)",
            data=csv_buf,
            file_name="transformed_data.csv",
            mime="text/csv",
            key="download_csv",
        )

except Exception as e:
    st.error(f"Error rendering plot: {e}")
    import traceback
    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════
# Step 5 — Send to AI Copilot
# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### 🧠 Analyze with AI Copilot")
st.caption("Describe what you see or ask a question. The AI will receive your full plot context.")

user_question = st.text_area(
    "Your question or observation",
    placeholder="e.g. 'Why does temperature spike around October?' or 'What complementary sensors should I compare?'",
    key="copilot_q",
    height=80,
)

if st.button("🧠 Send to Copilot", key="send_copilot", type="primary"):
    # Build a rich context string
    parts = [f"**Context from Analysis Workspace:**"]
    parts.append(f"- Dataset: `{data_label}`")
    parts.append(f"- Plot type: `{plot_type}`")
    parts.append(f"- Rows after transforms: {len(plot_df):,}")

    if plot_type in ("Line", "Scatter", "Depth Profile (inverted Y)"):
        parts.append(f"- X-axis: `{x_axis}`, Y-axis: `{y_axis}`")
        if color_col != "None":
            parts.append(f"- Color mapped to: `{color_col}`")
    elif plot_type == "Multi-Y Overlay":
        parts.append(f"- X-axis: `{x_axis}`, Y-axes: {y_axes}")
    elif plot_type == "Histogram":
        parts.append(f"- Variable: `{hist_col}`, Bins: {hist_bins}")
    elif plot_type == "Box Plot":
        parts.append(f"- Variables: {box_cols}")

    if do_smooth:
        parts.append(f"- Smoothing: rolling average, window={smooth_window}")
    if do_resample:
        parts.append(f"- Resampled to: {resample_label}")
    if do_outlier:
        parts.append(f"- Outlier removal: IQR × {iqr_factor}")

    # Add basic statistics for the plotted variable
    if plot_type in ("Line", "Scatter", "Depth Profile (inverted Y)"):
        yc = resolve_col(y_axis)
        if yc in plot_df.columns:
            desc = plot_df[yc].describe()
            parts.append(f"\n**Quick stats for {y_axis}:**")
            parts.append(f"  mean={desc['mean']:.4f}, std={desc['std']:.4f}, min={desc['min']:.4f}, max={desc['max']:.4f}")

    if user_question:
        parts.append(f"\n**User's question:** {user_question}")
    else:
        parts.append("\nPlease help me interpret this data. What patterns do you see, and what complementary datasets should I pull to form a hypothesis?")

    prompt = "\n".join(parts)
    st.session_state.pending_prompt = prompt
    st.switch_page("pages/1_Chat.py")
