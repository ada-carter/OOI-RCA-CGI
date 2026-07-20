"""
Test 03 — LLM Manager

Tests provider initialization, hot-swapping, and the Fireworks
message builder. Does NOT load actual models.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLLMManagerInit:
    def test_local_provider_init(self):
        """LLMManager initializes with local provider."""
        from backend.core.llm_manager import LLMManager
        mgr = LLMManager(
            model_config={"repo_id": "test", "filename": "test.gguf"},
            provider="local",
        )
        assert mgr.provider_name == "local"

    def test_fireworks_provider_init(self):
        """LLMManager initializes with fireworks provider."""
        from backend.core.llm_manager import LLMManager
        mgr = LLMManager(
            model_config={"repo_id": "test", "filename": "test.gguf"},
            provider="fireworks",
            fireworks_api_key="test_key",
            fireworks_model="accounts/fireworks/models/test-model",
        )
        assert mgr.provider_name == "fireworks"


class TestProviderSwapping:
    def test_toggle_local_to_fireworks(self):
        """Hot-swap from local to fireworks and back."""
        from backend.core.llm_manager import LLMManager
        mgr = LLMManager(
            model_config={"repo_id": "test", "filename": "test.gguf"},
            provider="local",
        )
        assert mgr.provider_name == "local"

        mgr.set_provider(
            provider="fireworks",
            fireworks_api_key="test_key",
            fireworks_model="accounts/fireworks/models/test",
        )
        assert mgr.provider_name == "fireworks"

        mgr.set_provider(provider="local")
        assert mgr.provider_name == "local"

    def test_model_property_none_in_testing(self):
        """In TESTING mode, model property returns None (no GPU load)."""
        from backend.core.llm_manager import LLMManager
        mgr = LLMManager(
            model_config={"repo_id": "test", "filename": "test.gguf"},
            provider="local",
        )
        # In test mode, local model won't load
        assert mgr.model is None


class TestFireworksMessageBuilding:
    def test_user_message_structure(self):
        """FireworksProvider builds system + user messages from a prompt."""
        from backend.core.llm_manager import FireworksProvider
        provider = FireworksProvider(api_key="test", model="test-model")

        messages = provider._build_messages("Hello world", is_raw=False)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello world"

    def test_raw_message_passthrough(self):
        """Raw prompts are passed through as-is in a user message."""
        from backend.core.llm_manager import FireworksProvider
        provider = FireworksProvider(api_key="test", model="test-model")

        messages = provider._build_messages("raw prompt text", is_raw=True)
        # Raw mode should still produce at least one message
        assert len(messages) >= 1


class TestMockLLMResponses:
    def test_basic_response(self, mock_llm):
        """Mock LLM returns expected basic response."""
        response = mock_llm.generate_response("Hello")
        assert "mocked response" in response

    def test_m2m_trigger(self, mock_llm):
        """Mock LLM returns M2M tag when triggered."""
        response = mock_llm.generate_response("mock_m2m")
        assert "<m2m_request" in response

    def test_search_trigger(self, mock_llm):
        """Mock LLM returns search_instruments tag when triggered."""
        response = mock_llm.generate_response("mock_search")
        assert "<search_instruments" in response

    def test_stream_yields(self, mock_llm):
        """Mock stream yields at least one chunk."""
        chunks = list(mock_llm.generate_response_stream("Hello"))
        assert len(chunks) > 0
