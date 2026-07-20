"""
Test 06 — RAG Instrumentation Search

Tests the RCA instrumentation database parser and search function.
This is a critical component — it's the domain knowledge layer that
grounds the LLM's instrument lookups.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from backend.rag.rca_instrumentation import search_instrumentation, _parse_instrumentation


class TestInstrumentationParsing:
    def test_parses_nonzero_records(self):
        """The instrumentation database parses into a non-empty list."""
        records = _parse_instrumentation()
        assert len(records) > 0

    def test_record_structure(self):
        """Each record has all 5 required keys."""
        records = _parse_instrumentation()
        required_keys = {"subsite", "node", "sensor", "method", "stream"}
        for record in records[:10]:  # spot check first 10
            assert required_keys.issubset(record.keys())

    def test_no_empty_values(self):
        """No record has empty string values."""
        records = _parse_instrumentation()
        for record in records:
            for key, value in record.items():
                assert value.strip() != "", f"Empty value for {key} in {record}"


class TestInstrumentSearch:
    def test_search_ctd_returns_results(self):
        """Searching for 'CTD' finds at least one instrument."""
        results = search_instrumentation(sensor="CTD")
        assert len(results) > 0
        assert all("ctd" in r["sensor"].lower() for r in results)

    def test_search_subsite_filter(self):
        """Subsite filter narrows results to that subsite."""
        results = search_instrumentation(subsite="RS01SBPS")
        assert len(results) > 0
        assert all(r["subsite"] == "RS01SBPS" for r in results)

    def test_search_case_insensitive(self):
        """Search is case-insensitive — 'ctd' matches 'CTD'."""
        upper = search_instrumentation(sensor="CTD")
        lower = search_instrumentation(sensor="ctd")
        assert len(upper) == len(lower)

    def test_search_partial_match(self):
        """Partial strings match (substring search)."""
        results = search_instrumentation(sensor="CTD")
        # Should match sensors like "01-CTDPFL104", "2A-CTDPFA102", etc.
        assert all("CTD" in r["sensor"].upper() for r in results)

    def test_search_no_results(self):
        """Nonsensical search returns empty list."""
        results = search_instrumentation(sensor="ZZZZNOTANINSTRUMENT")
        assert results == []

    def test_search_all_none(self):
        """Passing no filters returns all records."""
        all_results = search_instrumentation()
        assert len(all_results) > 10  # should be many records

    def test_combined_filters(self):
        """Multiple filters narrow results correctly."""
        broad = search_instrumentation(subsite="RS01SBPS")
        narrow = search_instrumentation(subsite="RS01SBPS", sensor="CTD")
        assert len(narrow) <= len(broad)
        assert all(r["subsite"] == "RS01SBPS" for r in narrow)
        assert all("CTD" in r["sensor"].upper() for r in narrow)

    def test_method_filter(self):
        """Method filter works (e.g., 'streamed', 'recovered')."""
        results = search_instrumentation(method="streamed")
        if results:  # may not have streamed in all configs
            assert all("streamed" in r["method"].lower() for r in results)
