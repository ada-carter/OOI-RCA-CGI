"""
Test 08 — Agent Loop Integration

Tests the full agent loop flow using mocked LLM responses.
Verifies event yielding, M2M detection, and prompt building.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_utils.chat_engine import build_prompt, agent_loop


class TestPromptBuilding:
    def test_single_turn(self):
        """Single user turn produces correct Gemma prompt format."""
        turns = [{"role": "user", "content": "Hello"}]
        prompt = build_prompt(turns, "You are a helpful assistant.")
        assert "<start_of_turn>user" in prompt
        assert "Hello" in prompt
        assert "<start_of_turn>model" in prompt
        assert "You are a helpful assistant." in prompt

    def test_multi_turn(self):
        """Multi-turn conversation preserves role ordering."""
        turns = [
            {"role": "user", "content": "What is a CTD?"},
            {"role": "model", "content": "A CTD measures conductivity, temperature, and depth."},
            {"role": "user", "content": "Where are they deployed?"},
        ]
        prompt = build_prompt(turns, "System prompt.")
        assert "What is a CTD?" in prompt
        assert "conductivity, temperature, and depth" in prompt
        assert "Where are they deployed?" in prompt

    def test_empty_turns(self):
        """Empty turns list still produces a valid prompt."""
        prompt = build_prompt([], "System prompt.")
        assert "<start_of_turn>model" in prompt


class TestAgentLoop:
    def test_basic_response_events(self, mock_llm):
        """Agent loop yields token events and a done event for basic input."""
        events = list(agent_loop("Hello", "test_project", mock_llm))

        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(token_events) > 0, "No tokens were yielded"
        assert len(done_events) == 1, "Missing done event"

        full_text = "".join(e["text"] for e in token_events)
        assert "mocked response" in full_text

    def test_m2m_detection(self, mock_llm):
        """Agent loop detects M2M request tags and yields m2m_request event."""
        events = list(agent_loop("mock_m2m", "test_project", mock_llm))

        m2m_events = [e for e in events if e["type"] == "m2m_request"]
        assert len(m2m_events) == 1

        params = m2m_events[0]["params"]
        assert params["subsite"] == "RS01SBPS"
        assert params["sensor"] == "2A-CTDPFA102"
        assert params["method"] == "streamed"
        assert params["begin_dt"] == "2024-01-01T00:00:00.000Z"
        assert params["end_dt"] == "2024-01-02T00:00:00.000Z"

    def test_done_event_is_final(self, mock_llm):
        """Done event is always the last event yielded."""
        events = list(agent_loop("Hello", "test_project", mock_llm))
        assert events[-1]["type"] == "done"

        # Full text is recoverable from token events
        token_events = [e for e in events if e["type"] == "token"]
        full_text = "".join(e["text"] for e in token_events)
        assert len(full_text) > 0
