"""
Test 05 — Data Processor

Tests the NetCDF processing pipeline: statistics, NTP time
conversion, QC flag masking, and export.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from backend.services.data_processor import DataProcessor, NTP_EPOCH_OFFSET


@pytest.fixture
def processor():
    return DataProcessor()


@pytest.fixture
def sample_dataset():
    """A minimal dataset mimicking OOI CTD data with NTP timestamps."""
    ntp_times = np.array([
        NTP_EPOCH_OFFSET + 1704067200,  # 2024-01-01 00:00:00 UTC
        NTP_EPOCH_OFFSET + 1704067260,  # + 1 minute
        NTP_EPOCH_OFFSET + 1704067320,  # + 2 minutes
        NTP_EPOCH_OFFSET + 1704067380,  # + 3 minutes
    ], dtype=np.float64)

    return xr.Dataset(
        {
            "sea_water_temperature": ("obs", [10.5, 10.6, 10.4, np.nan]),
            "sea_water_practical_salinity": ("obs", [34.1, 34.2, 34.0, 34.3]),
        },
        coords={"time": ("obs", ntp_times)},
    )


@pytest.fixture
def qc_dataset():
    """Dataset with QC flag variables for masking tests."""
    return xr.Dataset({
        "sea_water_temperature": ("obs", [10.5, 10.6, 10.4, 10.3, 10.7]),
        "sea_water_temperature_qc_results": ("obs", [1, 1, 3, 4, 1]),
        "sea_water_temperature_qc_executed": ("obs", [1, 1, 1, 1, 1]),
    })


# ═══════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════

class TestStatistics:
    def test_basic_stats(self, processor):
        """Computes correct mean, std, min, max for clean data."""
        ds = xr.Dataset({"temp": ("time", [1.0, 2.0, 3.0, 4.0, 5.0])})
        stats = processor.generate_statistics(ds, variables=["temp"])
        assert stats["temp"]["mean"] == pytest.approx(3.0)
        assert stats["temp"]["min"] == pytest.approx(1.0)
        assert stats["temp"]["max"] == pytest.approx(5.0)
        assert stats["temp"]["count"] == 5
        assert stats["temp"]["coverage_pct"] == 100.0

    def test_stats_with_nans(self, processor):
        """NaN values are excluded from stats but counted in coverage."""
        ds = xr.Dataset({"temp": ("time", [1.0, 2.0, 3.0, np.nan])})
        stats = processor.generate_statistics(ds, variables=["temp"])
        assert stats["temp"]["mean"] == pytest.approx(2.0)
        assert stats["temp"]["count"] == 3
        assert stats["temp"]["coverage_pct"] == 75.0

    def test_stats_all_nan(self, processor):
        """All-NaN variable returns None stats with 0% coverage."""
        ds = xr.Dataset({"temp": ("time", [np.nan, np.nan, np.nan])})
        stats = processor.generate_statistics(ds, variables=["temp"])
        assert stats["temp"]["mean"] is None
        assert stats["temp"]["count"] == 0
        assert stats["temp"]["coverage_pct"] == 0.0

    def test_auto_variable_selection(self, processor):
        """Auto-selects numeric variables, skips QC variables."""
        ds = xr.Dataset({
            "temperature": ("obs", [10.0, 11.0]),
            "temperature_qc_results": ("obs", [1, 1]),
            "temperature_qc_executed": ("obs", [1, 1]),
        })
        stats = processor.generate_statistics(ds)
        assert "temperature" in stats
        assert "temperature_qc_results" not in stats
        assert "temperature_qc_executed" not in stats

    def test_nonexistent_variable(self, processor):
        """Requesting stats for a variable not in dataset is silently skipped."""
        ds = xr.Dataset({"temp": ("time", [1.0, 2.0])})
        stats = processor.generate_statistics(ds, variables=["nonexistent"])
        assert "nonexistent" not in stats


# ═══════════════════════════════════════════════════════════════
# NTP Time Conversion
# ═══════════════════════════════════════════════════════════════

class TestNTPConversion:
    def test_converts_float_timestamps(self, processor, sample_dataset):
        """NTP float timestamps are converted to datetime64."""
        ds = processor.convert_ntp_to_datetime(sample_dataset)
        time_vals = ds["time"].values
        assert np.issubdtype(time_vals.dtype, np.datetime64)

    def test_already_datetime_unchanged(self, processor):
        """If time is already datetime64, no conversion occurs."""
        import pandas as pd
        ds = xr.Dataset({
            "time": ("obs", pd.to_datetime(["2024-01-01", "2024-01-02"])),
            "temp": ("obs", [10.0, 11.0]),
        })
        result = processor.convert_ntp_to_datetime(ds)
        assert np.issubdtype(result["time"].values.dtype, np.datetime64)

    def test_no_time_variable(self, processor):
        """Dataset without 'time' variable passes through unchanged."""
        ds = xr.Dataset({"temp": ("obs", [10.0, 11.0])})
        result = processor.convert_ntp_to_datetime(ds)
        assert "time" not in result


# ═══════════════════════════════════════════════════════════════
# QC Flag Masking
# ═══════════════════════════════════════════════════════════════

class TestQCMasking:
    def test_masks_bad_and_suspect(self, processor, qc_dataset):
        """Flags 3 (suspect) and 4 (fail) are masked to NaN."""
        ds, summary = processor.apply_qc_flags(qc_dataset)
        temp_vals = ds["sea_water_temperature"].values
        # Index 2 (flag=3) and index 3 (flag=4) should be NaN
        assert np.isnan(temp_vals[2])
        assert np.isnan(temp_vals[3])
        # Index 0, 1, 4 (flag=1) should be preserved
        assert temp_vals[0] == pytest.approx(10.5)
        assert temp_vals[4] == pytest.approx(10.7)

    def test_qc_summary_metrics(self, processor, qc_dataset):
        """QC summary reports correct drop counts."""
        _, summary = processor.apply_qc_flags(qc_dataset)
        assert "sea_water_temperature" in summary
        assert summary["sea_water_temperature"]["dropped"] == 2
        assert summary["sea_water_temperature"]["total"] == 5
        assert summary["sea_water_temperature"]["percent_dropped"] == 40.0

    def test_no_qc_vars(self, processor):
        """Dataset without QC variables passes through unchanged."""
        ds = xr.Dataset({"temp": ("obs", [10.0, 11.0])})
        result, summary = processor.apply_qc_flags(ds)
        assert summary == {}
        assert result["temp"].values.tolist() == [10.0, 11.0]


# ═══════════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════════

class TestExport:
    def test_export_csv(self, processor, tmp_path):
        """Exports dataset to CSV with correct row count."""
        ds = xr.Dataset({"temp": ("obs", [10.0, 11.0, 12.0])})
        out_path = str(tmp_path / "test_export.csv")
        result = processor.export_csv(ds, out_path)
        assert Path(result).exists()
        import pandas as pd
        df = pd.read_csv(result)
        assert len(df) == 3
        assert "temp" in df.columns
