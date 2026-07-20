"""
Shared test configuration and fixtures.

Sets TESTING=1 to prevent model loading, configures paths,
and provides reusable fixtures for the test suite.
"""

import os
os.environ["TESTING"] = "1"

import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

# Add the backend directory to the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR))


# ── LLM Mock ──────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """
    Mocks the LLMManager to return deterministic responses instead of loading the massive model.
    """
    from backend.core.llm_manager import LLMManager

    mock_manager = MagicMock(spec=LLMManager)
    mock_manager.provider_name = "local"

    def mock_generate(prompt, **kwargs):
        if "mock_m2m" in prompt:
            return (
                'Sure, let me request that data for you. '
                '<m2m_request subsite="RS01SBPS" node="SF01A" sensor="2A-CTDPFA102" '
                'method="streamed" stream="ctdpf_sbe43_sample" '
                'begin_dt="2024-01-01T00:00:00.000Z" end_dt="2024-01-02T00:00:00.000Z"/>'
            )
        if "mock_search" in prompt:
            return (
                'Let me search for CTD instruments. '
                '<search_instruments sensor="CTD"/>'
            )
        return "This is a mocked response from the LLM."

    def mock_generate_stream(prompt, **kwargs):
        response = mock_generate(prompt)
        yield response

    def mock_tokenize(text):
        return list(range(len(text) // 4))

    mock_manager.generate_response.side_effect = mock_generate
    mock_manager.generate_response_stream.side_effect = mock_generate_stream
    mock_manager.tokenize.side_effect = mock_tokenize
    mock_manager.model = None

    return mock_manager


# ── Isolated DB + Project Manager ─────────────────────────────

@pytest.fixture
def tmp_project_dir(tmp_path):
    """Provides a clean temporary directory for project data."""
    project_dir = tmp_path / "test_projects"
    project_dir.mkdir()
    return project_dir


@pytest.fixture
def isolated_db(tmp_path):
    """
    Creates a temporary SQLite database for testing, completely
    isolated from the production aida.db.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db.database import Base
    # Import models so their tables are registered on Base.metadata
    import db.models  # noqa: F401

    db_path = tmp_path / "test_aida.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return TestSession


@pytest.fixture
def project_manager(isolated_db, tmp_project_dir, monkeypatch):
    """
    Provides a ProjectManager backed by a temp DB and temp directory.
    Monkey-patches SessionLocal so no production data is touched.
    """
    # Patch at every import site that uses SessionLocal
    import db.database as db_mod
    monkeypatch.setattr(db_mod, "SessionLocal", isolated_db)

    # Also reimport project_manager module so it picks up the patched SessionLocal
    from backend.state import project_manager as pm_mod
    monkeypatch.setattr(pm_mod, "SessionLocal", isolated_db)

    from backend.state.project_manager import ProjectManager
    pm = ProjectManager()
    pm.base_dir = tmp_project_dir
    return pm
