"""
OOI Machine-to-Machine (M2M) REST API Client.

Provides authenticated access to the OOI data catalog:
  - Browse subsites, nodes, sensors, methods, and streams
  - Submit asynchronous data requests
  - Poll THREDDS catalog for file availability
  - Download NetCDF files with retry logic

Reference: https://ooinet.oceanobservatories.org/api/m2m/
"""

import logging
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv"


class M2MClient:
    """Client for the OOI M2M REST API with retry + rate-limit handling."""

    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.session = requests.Session()
        self.session.auth = (username, token)
        self.session.headers.update({"Accept": "application/json"})

        # Retry strategy for transient errors
        retry = Retry(
            total=4,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ──────────────────────────────────────────────────────
    # Browse the instrument hierarchy
    # ──────────────────────────────────────────────────────

    def _get_json(self, url: str, params: Optional[dict] = None) -> Any:
        """GET helper with rate-limit back-off."""
        resp = self.session.get(url, params=params, timeout=30, auth=(self.username, self.token))
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            logger.warning(f"Rate-limited by OOI API. Waiting {wait}s...")
            time.sleep(wait)
            resp = self.session.get(url, params=params, timeout=30, auth=(self.username, self.token))
        resp.raise_for_status()
        return resp.json()

    def list_sites(self) -> List[str]:
        """Return all available subsites (e.g. ['RS01SBPS', ...])."""
        return self._get_json(BASE_URL)

    def list_nodes(self, subsite: str) -> List[str]:
        """Return nodes under a subsite (e.g. ['SF01A', ...])."""
        return self._get_json(f"{BASE_URL}/{subsite}")

    def list_sensors(self, subsite: str, node: str) -> List[str]:
        """Return sensors under a node (e.g. ['2A-CTDPFA102', ...])."""
        return self._get_json(f"{BASE_URL}/{subsite}/{node}")

    def list_methods(self, subsite: str, node: str, sensor: str) -> List[str]:
        """Return delivery methods (e.g. ['streamed', 'recovered_inst'])."""
        return self._get_json(f"{BASE_URL}/{subsite}/{node}/{sensor}")

    def list_streams(
        self, subsite: str, node: str, sensor: str, method: str
    ) -> List[str]:
        """Return data streams for a given method."""
        return self._get_json(f"{BASE_URL}/{subsite}/{node}/{sensor}/{method}")

    # ──────────────────────────────────────────────────────
    # Submit async data requests
    # ──────────────────────────────────────────────────────

    def request_data(
        self,
        subsite: str,
        node: str,
        sensor: str,
        method: str,
        stream: str,
        begin_dt: str,
        end_dt: str,
        data_format: str = "application/netcdf",
    ) -> Dict[str, Any]:
        """
        Submit an asynchronous data request and return the response
        containing `requestUUID` and `allURLs`.

        Parameters
        ----------
        begin_dt : ISO-8601 string, e.g. '2024-01-01T00:00:00.000Z'
        end_dt   : ISO-8601 string
        data_format : 'application/netcdf' or 'text/csv'
        """
        url = f"{BASE_URL}/{subsite}/{node}/{sensor}/{method}/{stream}"
        params = {
            "beginDT": begin_dt,
            "endDT": end_dt,
            "format": data_format,
        }
        logger.info(f"Requesting data: {url} | {params}")
        resp = self.session.get(url, params=params, timeout=60, auth=(self.username, self.token))
        
        # Intercept OOI API specific errors
        if not resp.ok:
            try:
                err_data = resp.json()
                if isinstance(err_data.get("message"), dict):
                    msg = err_data["message"].get("status", "Unknown API error")
                    raise ValueError(f"OOI API Error: {msg}")
                elif "message" in err_data:
                    raise ValueError(f"OOI API Error: {err_data['message']}")
            except ValueError:
                raise  # Re-raise the parsed OOI error
            except Exception:
                pass   # Fall back to raise_for_status if JSON parsing fails

        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Request UUID: {data.get('requestUUID')}")
        return data

    # ──────────────────────────────────────────────────────
    # Poll THREDDS & download
    # ──────────────────────────────────────────────────────

    def get_thredds_url(self, request_response: Dict[str, Any]) -> Optional[str]:
        """
        Extract the THREDDS catalog URL from a request_data() response.
        The THREDDS URL is the one containing 'thredds' in allURLs.
        """
        urls = request_response.get("allURLs", [])
        for url in urls:
            if "thredds" in url.lower():
                return url
        return urls[0] if urls else None

    def check_thredds_status(
        self, thredds_url: str, max_wait: int = 600, poll_interval: int = 15
    ) -> List[str]:
        """
        Poll the THREDDS catalog directory until .nc files appear.

        Returns a list of direct download URLs for .nc files.
        Raises TimeoutError if files don't appear within max_wait seconds.
        """
        logger.info(f"Polling THREDDS catalog: {thredds_url}")
        elapsed = 0
        while elapsed < max_wait:
            try:
                resp = self.session.get(thredds_url, timeout=30)
                if resp.status_code == 200:
                    nc_links = self._parse_thredds_catalog(resp.text, thredds_url)
                    if nc_links:
                        logger.info(f"Found {len(nc_links)} NetCDF file(s) on THREDDS.")
                        return nc_links
            except requests.RequestException as e:
                logger.warning(f"THREDDS poll error: {e}")

            logger.info(
                f"Data not ready yet. Waiting {poll_interval}s... "
                f"({elapsed}/{max_wait}s elapsed)"
            )
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(
            f"THREDDS data not ready after {max_wait}s. URL: {thredds_url}"
        )

    def _parse_thredds_catalog(self, html: str, base_url: str) -> List[str]:
        """
        Parse the THREDDS directory listing HTML to extract .nc file URLs.
        """
        nc_files = []
        
        # OOI TDS often uses catalog.html?dataset=DATASET_ID format
        pattern1 = re.compile(r'href=["\']catalog\.html\?dataset=([^"\']+\.nc)["\']', re.IGNORECASE)
        for match in pattern1.finditer(html):
            dataset_id = match.group(1)
            # The download URL is simply /thredds/fileServer/ + dataset_id
            domain = base_url.split('/thredds/')[0]
            nc_files.append(f"{domain}/thredds/fileServer/{dataset_id}")
            
        if not nc_files:
            # Fallback for standard THREDDS
            pattern2 = re.compile(r'href=["\']([^"\']*\.nc)["\']', re.IGNORECASE)
            for match in pattern2.finditer(html):
                href = match.group(1)
                if href.startswith("http"):
                    nc_files.append(href)
                else:
                    # Switch from catalog HTML to fileServer download path
                    download_base = base_url.replace('/thredds/catalog/', '/thredds/fileServer/')
                    if download_base.endswith("/"):
                        nc_files.append(download_base + href)
                    else:
                        nc_files.append(download_base.rsplit("/", 1)[0] + "/" + href)
        return nc_files

    def download_netcdf_files(
        self,
        file_urls: List[str],
        dest_dir: str,
        max_retries: int = 3,
    ) -> List[str]:
        """
        Download .nc files from THREDDS to dest_dir.
        Returns list of local file paths.
        """
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        downloaded = []

        for url in file_urls:
            filename = url.rsplit("/", 1)[-1]
            local_path = dest_path / filename
            logger.info(f"Downloading: {filename}")

            for attempt in range(1, max_retries + 1):
                try:
                    resp = self.session.get(url, stream=True, timeout=120)
                    resp.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Verify non-zero size
                    if local_path.stat().st_size > 0:
                        downloaded.append(str(local_path))
                        logger.info(f"  Saved: {local_path}")
                        break
                    else:
                        logger.warning(f"  Empty file on attempt {attempt}.")
                        local_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(
                        f"  Download attempt {attempt}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)

        return downloaded

    # ──────────────────────────────────────────────────────
    # Convenience: full pipeline
    # ──────────────────────────────────────────────────────

    def request_and_download(
        self,
        subsite: str,
        node: str,
        sensor: str,
        method: str,
        stream: str,
        begin_dt: str,
        end_dt: str,
        dest_dir: str,
        data_format: str = "application/netcdf",
        max_wait: int = 600,
    ) -> List[str]:
        """
        End-to-end: submit request → poll THREDDS → download files.
        Returns list of downloaded local file paths.
        """
        response = self.request_data(
            subsite, node, sensor, method, stream, begin_dt, end_dt, data_format
        )
        thredds_url = self.get_thredds_url(response)
        if not thredds_url:
            raise ValueError("No THREDDS URL found in OOI response.")

        file_urls = self.check_thredds_status(thredds_url, max_wait=max_wait)
        return self.download_netcdf_files(file_urls, dest_dir)

