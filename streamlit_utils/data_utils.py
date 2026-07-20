import io
import pandas as pd
import xarray as xr
from pathlib import Path
import streamlit as st

@st.cache_resource(show_spinner="Loading dataset…")
def load_csv(path_or_bytes, filename: str = "upload"):
    """
    Load a CSV from a file path (str/Path) or an UploadedFile bytes object.
    Uses @st.cache_resource to return a memory reference instead of deeply 
    copying (which st.cache_data does) to prevent massive UI latency on large datasets.
    """
    if isinstance(path_or_bytes, (str, Path)):
        df = pd.read_csv(path_or_bytes)
    else:
        df = pd.read_csv(io.BytesIO(path_or_bytes))

    # Auto-parse any column that looks like a timestamp
    for col in df.columns:
        if "time" in col.lower() or "timestamp" in col.lower():
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:  # at least half parsed OK
                    df[col] = parsed
            except Exception:
                pass
    return df

@st.cache_resource(show_spinner="Loading NetCDF dataset…")
def load_netcdf(path_or_bytes):
    """
    Load a NetCDF file natively using xarray.
    Returns a lazy-loaded xr.Dataset reference.
    """
    if isinstance(path_or_bytes, (str, Path)):
        return xr.open_dataset(path_or_bytes, engine='netcdf4')
    else:
        # For uploaded bytes, we must write to a temp file because xarray requires a file path or h5netcdf
        import tempfile
        import os
        fd, temp_path = tempfile.mkstemp(suffix=".nc")
        with os.fdopen(fd, 'wb') as f:
            f.write(path_or_bytes)
        ds = xr.open_dataset(temp_path, engine='netcdf4')
        # Note: We don't delete the temp file immediately because it's lazy-loaded. 
        # For a robust solution, we'd manage temp file cleanup separately.
        return ds
