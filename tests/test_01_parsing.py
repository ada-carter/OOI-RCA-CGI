"""
Test 01 — Tag Parsing

Tests all LLM output tag parsers: thoughts, M2M requests,
search_instruments, update_view, generate_plot, render_map.
These are the safety-critical parsing functions that sit between
the LLM and actual system actions.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_utils.chat_engine import (
    parse_thoughts,
    parse_m2m_request,
    parse_search_instruments,
    parse_update_view,
    parse_plot_requests,
    parse_map_requests,
)


# ═══════════════════════════════════════════════════════════════
# Thought / Channel Token Parsing
# ═══════════════════════════════════════════════════════════════

class TestThoughtParsing:
    def test_strips_channel_thought_tags(self):
        """Gemma channel+thought tokens are extracted and cleaned."""
        text = "<|channel>thought\nSome internal reasoning\n<channel|>The actual response."
        thoughts, response = parse_thoughts(text)
        assert "internal reasoning" in thoughts
        assert "actual response" in response
        assert "<channel" not in response

    def test_plain_text_passthrough(self):
        """Text without thought tokens passes through unmodified."""
        text = "This is a normal response with no thought tokens."
        thoughts, response = parse_thoughts(text)
        assert thoughts == ""
        assert response == text

    def test_strips_orphan_channel_tags(self):
        """Leftover channel tags are cleaned even without paired thought block."""
        text = "<|channel>text\nHello world"
        thoughts, response = parse_thoughts(text)
        assert "<|channel>" not in response
        assert "<channel|>" not in response

    def test_preserves_markdown(self):
        """Standard markdown formatting is preserved after parsing."""
        text = "# Heading\n**bold text**\n- list item"
        thoughts, response = parse_thoughts(text)
        assert "# Heading" in response
        assert "**bold text**" in response

    def test_strips_think_tags(self):
        """<think>...</think> tags (DeepSeek style) are cleaned."""
        text = "<think>internal reasoning</think>The actual response."
        thoughts, response = parse_thoughts(text)
        assert "<think>" not in response
        assert "</think>" not in response


# ═══════════════════════════════════════════════════════════════
# M2M Request Parsing
# ═══════════════════════════════════════════════════════════════

class TestM2MParsing:
    def test_valid_complete_tag(self):
        """All 7 required attributes are extracted from a well-formed tag."""
        text = '''Here is some text.
        <m2m_request subsite="RS01SBPS" node="SF01A" sensor="2A-CTDPFA102"
        method="streamed" stream="ctdpf_sbe43_sample"
        begin_dt="2024-01-01T00:00:00.000Z" end_dt="2024-01-02T00:00:00.000Z"/>
        And some more text.'''

        result = parse_m2m_request(text)
        assert result is not None
        assert result["subsite"] == "RS01SBPS"
        assert result["node"] == "SF01A"
        assert result["sensor"] == "2A-CTDPFA102"
        assert result["method"] == "streamed"
        assert result["stream"] == "ctdpf_sbe43_sample"
        assert result["begin_dt"] == "2024-01-01T00:00:00.000Z"
        assert result["end_dt"] == "2024-01-02T00:00:00.000Z"

    def test_returns_none_on_missing_attrs(self):
        """Incomplete tags (missing required attrs) return None."""
        text = '<m2m_request subsite="RS01SBPS" node="SF01A"/>'
        result = parse_m2m_request(text)
        assert result is None

    def test_returns_none_on_no_tag(self):
        """Plain text with no M2M tag returns None."""
        result = parse_m2m_request("Just some regular text about data.")
        assert result is None

    def test_md2m_typo_variant(self):
        """The LLM sometimes outputs <md2m_request> — parser handles it."""
        text = '<md2m_request subsite="RS01SBPS" node="SF01A" sensor="2A-CTDPFA102" method="streamed" stream="ctdpf_sbe43_sample" begin_dt="2024-01-01T00:00:00.000Z" end_dt="2024-01-02T00:00:00.000Z"/>'
        result = parse_m2m_request(text)
        assert result is not None
        assert result["subsite"] == "RS01SBPS"

    def test_multiline_tag(self):
        """Tag split across multiple lines is still parsed."""
        text = '''<m2m_request
            subsite="CE04OSBP"
            node="LJ01C"
            sensor="06-CTDBPO108"
            method="streamed"
            stream="ctdbp_no_sample"
            begin_dt="2023-06-01T00:00:00.000Z"
            end_dt="2023-06-02T00:00:00.000Z"
        />'''
        result = parse_m2m_request(text)
        assert result is not None
        assert result["subsite"] == "CE04OSBP"


# ═══════════════════════════════════════════════════════════════
# Search Instruments Parsing
# ═══════════════════════════════════════════════════════════════

class TestSearchInstrumentsParsing:
    def test_sensor_only(self):
        """Search with just sensor attribute."""
        text = '<search_instruments sensor="CTD"/>'
        result = parse_search_instruments(text)
        assert result is not None
        assert result["sensor"] == "CTD"

    def test_subsite_and_node(self):
        """Search with subsite and node."""
        text = '<search_instruments subsite="RS01SBPS" node="SF01A"/>'
        result = parse_search_instruments(text)
        assert result["subsite"] == "RS01SBPS"
        assert result["node"] == "SF01A"

    def test_all_attributes(self):
        """Search with all four attributes."""
        text = '<search_instruments subsite="RS01SBPS" node="SF01A" sensor="CTD" method="streamed"/>'
        result = parse_search_instruments(text)
        assert len(result) == 4

    def test_empty_search(self):
        """Empty tag (match all) returns empty dict, not None."""
        text = '<search_instruments />'
        result = parse_search_instruments(text)
        assert result is not None
        assert result == {}

    def test_no_tag_returns_none(self):
        """No search tag returns None."""
        result = parse_search_instruments("Just text, no search tag.")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# Update View Parsing
# ═══════════════════════════════════════════════════════════════

class TestUpdateViewParsing:
    def test_all_attributes(self):
        """Parses plot, dataset, and flowchart from update_view tag."""
        text = '<update_view plot="projects/test/plots/fig.png" dataset="projects/test/data.csv" flowchart="Download -> Clean -> Plot"/>'
        result = parse_update_view(text)
        assert result is not None
        assert result["plot"] == "projects/test/plots/fig.png"
        assert result["dataset"] == "projects/test/data.csv"
        assert result["flowchart"] == "Download -> Clean -> Plot"

    def test_partial_attributes(self):
        """Parses correctly when only plot is provided."""
        text = '<update_view plot="fig.png"/>'
        result = parse_update_view(text)
        assert result is not None
        assert result["plot"] == "fig.png"
        assert "dataset" not in result

    def test_no_tag(self):
        result = parse_update_view("No update view here.")
        assert result is None


# ═══════════════════════════════════════════════════════════════
# Plot & Map Request Parsing
# ═══════════════════════════════════════════════════════════════

class TestPlotParsing:
    def test_valid_plot_request(self):
        """Minimum required attributes parse correctly."""
        text = '<generate_plot dataset="data.csv" x_col="time" y_col="temperature" title="CTD Temp"/>'
        results = parse_plot_requests(text)
        assert len(results) == 1
        assert results[0]["dataset"] == "data.csv"
        assert results[0]["y_col"] == "temperature"

    def test_missing_required_attrs(self):
        """Plot tag without required attrs is skipped."""
        text = '<generate_plot dataset="data.csv"/>'
        results = parse_plot_requests(text)
        assert len(results) == 0

    def test_multiple_plot_tags(self):
        """Multiple plot tags in one response are all parsed."""
        text = (
            '<generate_plot dataset="d1.csv" x_col="time" y_col="temp" title="T"/>'
            '<generate_plot dataset="d2.csv" x_col="time" y_col="salinity" title="S"/>'
        )
        results = parse_plot_requests(text)
        assert len(results) == 2


class TestMapParsing:
    def test_valid_map_request(self):
        text = '<render_map lat="44.569" lon="-125.149" title="Slope Base"/>'
        results = parse_map_requests(text)
        assert len(results) == 1
        assert results[0]["lat"] == "44.569"
        assert results[0]["lon"] == "-125.149"

    def test_map_without_title(self):
        """Map tag without title still parses (title is optional)."""
        text = '<render_map lat="44.569" lon="-125.149"/>'
        results = parse_map_requests(text)
        assert len(results) == 1

    def test_no_map_tag(self):
        results = parse_map_requests("No map here.")
        assert len(results) == 0
