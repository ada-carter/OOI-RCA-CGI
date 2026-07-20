"""
Parses the RCA instrumentation Markdown file and provides a search function
that the agent's generated scripts can import and call.

Usage (from a script executed by the agent loop):
    from backend.rag.rca_instrumentation import search_instrumentation
    results = search_instrumentation(subsite=None, node=None, sensor=None, method=None)
"""

import re
from pathlib import Path
from typing import List, Dict, Optional

_MD_PATH = Path(__file__).with_name("rca_instrumentation.md")


def _parse_instrumentation(md_path: Path = _MD_PATH) -> List[Dict[str, str]]:
    """Parse the Markdown file into a flat list of endpoint records.

    Each record is a dict with keys:
        subsite, node, sensor, method, stream
    """
    text = md_path.read_text(encoding="utf-8")
    records: List[Dict[str, str]] = []

    current_subsite: Optional[str] = None
    current_node: Optional[str] = None
    current_sensor: Optional[str] = None
    current_method: Optional[str] = None

    for line in text.splitlines():
        stripped = line.strip()

        # ## Subsite: `RS01SBPD`
        m = re.match(r"^##\s+Subsite:\s+`([^`]+)`", stripped)
        if m:
            current_subsite = m.group(1)
            current_node = None
            current_sensor = None
            current_method = None
            continue

        # ### Node: `PD01A`
        m = re.match(r"^###\s+Node:\s+`([^`]+)`", stripped)
        if m:
            current_node = m.group(1)
            current_sensor = None
            current_method = None
            continue

        # - **Sensor**: `01-CTDPFL104`
        m = re.match(r"^-\s+\*\*Sensor\*\*:\s+`([^`]+)`", stripped)
        if m:
            current_sensor = m.group(1)
            current_method = None
            continue

        # - Method: `recovered_wfp`
        m = re.match(r"^-\s+Method:\s+`([^`]+)`", stripped)
        if m:
            current_method = m.group(1)
            continue

        # - Stream: `dpc_ctd_instrument_recovered`
        m = re.match(r"^-\s+Stream:\s+`([^`]+)`", stripped)
        if m:
            stream = m.group(1)
            if current_subsite and current_node and current_sensor and current_method:
                records.append({
                    "subsite": current_subsite,
                    "node": current_node,
                    "sensor": current_sensor,
                    "method": current_method,
                    "stream": stream,
                })
            continue

    return records


# Cache parsed records on first call
_CACHE: Optional[List[Dict[str, str]]] = None


def search_instrumentation(
    subsite: Optional[str] = None,
    node: Optional[str] = None,
    sensor: Optional[str] = None,
    method: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Search the RCA instrumentation database.

    Pass ``None`` for any parameter to match all values for that field.
    All comparisons are case-insensitive substring matches so that partial
    values like ``"CTD"`` or ``"ctd"`` still work.

    Returns a list of dicts, each with keys:
        subsite, node, sensor, method, stream
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _parse_instrumentation()

    results = _CACHE

    filters = {
        "subsite": subsite,
        "node": node,
        "sensor": sensor,
        "method": method,
    }

    for key, value in filters.items():
        if value is not None:
            value_lower = value.lower()
            results = [r for r in results if value_lower in r[key].lower()]

    return results
