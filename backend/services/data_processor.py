"""
NetCDF Data Processing Pipeline.

Handles loading, concatenation, QC masking, NTP time conversion,
statistical summaries, and export for OOI RCA data.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.variables import SerializationWarning

warnings.filterwarnings("ignore", category=SerializationWarning)

logger = logging.getLogger(__name__)

# OOI uses NTP epoch: seconds since 1900-01-01 00:00:00 UTC
# Delta to UNIX epoch (1970-01-01) in seconds
NTP_EPOCH_OFFSET = 2208988800


class DataProcessor:
    """Xarray/Pandas pipeline for OOI NetCDF datasets."""

    def __init__(self):
        pass

    # ──────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────

    def load_netcdf(self, file_path: str) -> xr.Dataset:
        """Load a single NetCDF file."""
        logger.info(f"Loading NetCDF: {file_path}")
        ds = xr.open_dataset(file_path, engine="netcdf4")
        return ds

    def concatenate_netcdf(self, file_paths: List[str]) -> xr.Dataset:
        """
        Merge multiple NetCDF files along the time dimension.
        Uses open_mfdataset for efficient lazy loading.
        """
        if not file_paths:
            raise ValueError("No file paths provided for concatenation.")

        logger.info(f"Concatenating {len(file_paths)} NetCDF file(s)...")
        def preprocess(ds):
            # Reset all coords except 'obs' and 'time' to data variables 
            # so open_mfdataset doesn't fail if they are missing in some files.
            coords_to_reset = [c for c in ds.coords if c not in ["obs", "time", "lat", "lon"]]
            return ds.reset_coords(coords_to_reset)

        ds = xr.open_mfdataset(
            file_paths,
            combine="nested",
            concat_dim="obs",  # OOI often uses 'obs' as the record dim
            engine="netcdf4",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            preprocess=preprocess,
        )
        logger.info(
            f"Concatenated dataset: {len(ds.data_vars)} variables, "
            f"{dict(ds.sizes)} dimensions"
        )
        return ds

    # ──────────────────────────────────────────────────────
    # Time Conversion
    # ──────────────────────────────────────────────────────

    def convert_ntp_to_datetime(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Convert OOI NTP epoch timestamps to standard datetime64.
        Looks for 'time' variable; if it's float/int, applies offset.
        """
        ds = dataset.copy()
        if "time" in ds:
            time_vals = ds["time"].values
            if np.issubdtype(time_vals.dtype, np.floating) or np.issubdtype(
                time_vals.dtype, np.integer
            ):
                logger.info("Converting NTP timestamps to datetime64...")
                unix_seconds = time_vals - NTP_EPOCH_OFFSET
                ds["time"] = pd.to_datetime(unix_seconds, unit="s", origin="unix")
                logger.info(
                    f"Time range: {ds['time'].values[0]} → {ds['time'].values[-1]}"
                )
        return ds

    # ──────────────────────────────────────────────────────
    # QA/QC
    # ──────────────────────────────────────────────────────

    def apply_qc_flags(self, dataset: xr.Dataset) -> tuple[xr.Dataset, Dict[str, Dict]]:
        """
        Mask data where QC provenance flags indicate 'bad' or 'suspect'.
        OOI convention: *_qc_executed and *_qc_results variables.
        Bit-pattern checking per QARTOD standards.
        Returns the masked dataset and a dictionary of QC summary metrics.
        """
        ds = dataset.copy()
        qc_vars = [v for v in ds.data_vars if v.endswith("_qc_results")]
        qc_summary = {}

        for qc_var in qc_vars:
            # Corresponding science variable
            sci_var = qc_var.replace("_qc_results", "")
            if sci_var not in ds:
                continue

            qc_values = ds[qc_var].values
            # QARTOD flag convention:
            #   1 = pass, 2 = not evaluated, 3 = suspect, 4 = fail, 9 = missing
            # Mask where any bit indicates fail (4) or suspect (3)
            # OOI stores as bitmask integers; check if bit patterns indicate bad
            bad_mask = np.isin(qc_values, [3, 4, 9])
            if bad_mask.any():
                count = int(bad_mask.sum())
                total = int(bad_mask.size)
                pct = count / total * 100
                
                qc_summary[sci_var] = {
                    "dropped": count,
                    "total": total,
                    "percent_dropped": round(pct, 2)
                }
                
                logger.info(
                    f"QC masking {sci_var}: {count}/{total} "
                    f"({pct:.1f}%) flagged as suspect/bad."
                )
                ds[sci_var] = ds[sci_var].where(~bad_mask)

        return ds, qc_summary

    # ──────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────

    def generate_statistics(
        self, dataset: xr.Dataset, variables: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for selected variables.
        Returns dict of {var_name: {mean, std, min, max, count, coverage_pct}}.
        """
        if variables is None:
            # Auto-select numeric science variables (skip QC, coords)
            variables = [
                v
                for v in dataset.data_vars
                if np.issubdtype(dataset[v].dtype, np.number)
                and not v.endswith(("_qc_results", "_qc_executed"))
            ]

        stats = {}
        for var in variables:
            if var not in dataset:
                continue
            arr = dataset[var].values.flatten()
            total = len(arr)
            valid = np.count_nonzero(~np.isnan(arr))
            coverage = (valid / total * 100) if total > 0 else 0.0

            if valid > 0:
                stats[var] = {
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "count": valid,
                    "total": total,
                    "coverage_pct": round(coverage, 2),
                }
            else:
                stats[var] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                    "total": total,
                    "coverage_pct": 0.0,
                }

        return stats

    # ──────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────

    def export_csv(self, dataset: xr.Dataset, output_path: str) -> str:
        """Flatten dataset to a tidy DataFrame and export as CSV."""
        logger.info(f"Exporting to CSV: {output_path}")
        df = dataset.to_dataframe().reset_index()
        df.to_csv(output_path, index=False)
        logger.info(f"CSV written: {len(df)} rows × {len(df.columns)} columns")
        return output_path

    def export_netcdf(self, dataset: xr.Dataset, output_path: str) -> str:
        """Save cleaned dataset back to NetCDF."""
        logger.info(f"Exporting to NetCDF: {output_path}")
        dataset.to_netcdf(output_path)
        logger.info(f"NetCDF written: {output_path}")
        return output_path

    # ──────────────────────────────────────────────────────
    # Full pipeline
    # ──────────────────────────────────────────────────────

    def process_pipeline(
        self,
        file_paths: List[str],
        output_dir: str,
        variables: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run the full processing pipeline:
        1. Concatenate NetCDF files
        2. Convert NTP timestamps
        3. Apply QC flags
        4. Generate statistics
        5. Export to processed NetCDF + CSV

        Returns dict with stats and output file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Concatenate
        ds = self.concatenate_netcdf(file_paths)

        # 2. Time conversion
        ds = self.convert_ntp_to_datetime(ds)

        # 3. QC masking
        ds, qc_summary = self.apply_qc_flags(ds)

        # 4. Statistics
        stats = self.generate_statistics(ds, variables)

        # 5. Export
        nc_out = str(output_path / "processed.nc")
        qc_out = output_path / "qc_summary.json"
        
        self.export_netcdf(ds, nc_out)
        
        import json
        qc_out.write_text(json.dumps(qc_summary, indent=2))

        return {
            "statistics": stats,
            "qc_summary": qc_summary,
            "netcdf_path": nc_out,
            "dimensions": dict(ds.sizes),
            "variables": list(ds.data_vars),
        }
