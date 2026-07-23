"""
OOI Cloud Zarr Client — fast-path data access.

Reads analysis-ready datasets directly from the public `ooi-data` S3 bucket of
consolidated Zarr stores, bypassing the asynchronous M2M request → THREDDS poll
→ NetCDF download cycle. When a stream is published to the cloud store, data is
available in seconds via lazy, chunked xarray access.

Dataset naming convention (ooi-data bucket):
    {subsite}-{node}-{sensor}-{method}-{stream}
e.g. RS01SBPS-SF01A-2A-CTDPFA102-streamed-ctdpf_sbe43_sample
(the reference designator subsite-node-sensor, plus method and stream — exactly
the 5 identifiers the agent already produces for an M2M request.)

NOTE FOR REVIEW: the bucket name, key convention, and `consolidated` flag are
the best-known public values for the OOI cloud store. Verify them against the
live bucket before relying on availability results. Every lookup degrades
gracefully (False / raised error caught upstream) so the M2M path stays the
fallback whenever the fast path is unavailable.
"""

import logging
from typing import Optional

import xarray as xr

logger = logging.getLogger(__name__)

# Public, anonymously-readable bucket of OOI cloud Zarr stores.
OOI_ZARR_BUCKET = "ooi-data"


class ZarrClient:
    """Read RCA data directly from the OOI cloud Zarr store."""

    def __init__(self, bucket: str = OOI_ZARR_BUCKET):
        self.bucket = bucket

    @staticmethod
    def build_dataset_id(subsite, node, sensor, method, stream) -> str:
        """Reference designator + method + stream, joined per ooi-data convention."""
        return f"{subsite}-{node}-{sensor}-{method}-{stream}"

    def dataset_exists(self, subsite, node, sensor, method, stream) -> bool:
        """Return True if a cloud Zarr store exists for this endpoint.

        Cheap existence check — probes for the store's metadata key rather than
        opening it. Any failure (s3fs missing, no network, no store) returns
        False so the caller falls back to M2M.
        """
        dataset_id = self.build_dataset_id(subsite, node, sensor, method, stream)
        try:
            import s3fs
            fs = s3fs.S3FileSystem(anon=True)
            prefix = f"{self.bucket}/{dataset_id}"
            return fs.exists(f"{prefix}/.zmetadata") or fs.exists(f"{prefix}/.zgroup")
        except Exception as e:
            logger.info(f"Zarr existence check failed for {dataset_id}: {e}")
            return False

    def open_dataset(
        self,
        subsite,
        node,
        sensor,
        method,
        stream,
        begin_dt: Optional[str] = None,
        end_dt: Optional[str] = None,
    ) -> xr.Dataset:
        """Open the cloud Zarr store and optionally subset by time.

        Returns a lazily-loaded dataset; the time slice reads only the chunks in
        range, so there is no full-file download.
        """
        import s3fs

        dataset_id = self.build_dataset_id(subsite, node, sensor, method, stream)
        fs = s3fs.S3FileSystem(anon=True)
        mapper = fs.get_mapper(f"{self.bucket}/{dataset_id}")

        try:
            ds = xr.open_zarr(mapper, consolidated=True)
        except Exception:
            ds = xr.open_zarr(mapper, consolidated=False)

        if "time" in ds.coords and (begin_dt or end_dt):
            ds = ds.sel(time=slice(begin_dt, end_dt))

        logger.info(f"Opened cloud Zarr store {dataset_id}: {dict(ds.sizes)}")
        return ds
