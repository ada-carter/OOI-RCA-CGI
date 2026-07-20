"""
Test 07 — Project Manager & Database

Tests project CRUD, chat history persistence, and directory
structure creation using an isolated temporary database.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestProjectCRUD:
    def test_create_project(self, project_manager):
        """Creating a project returns an ID and creates subdirectories."""
        pid = project_manager.create_project("user_1", "Test Project", "Testing purposes")
        path = project_manager.get_project_dir(pid)
        assert path.exists()
        assert (path / "raw_data").exists()
        assert (path / "processed_data").exists()
        assert (path / "scripts").exists()
        assert (path / "plots").exists()
        assert (path / "workflows").exists()

    def test_list_projects(self, project_manager):
        """Created project appears in user's project list."""
        project_manager.create_project("user_1", "Project A", "A")
        project_manager.create_project("user_1", "Project B", "B")
        projects = project_manager.list_projects("user_1")
        titles = [p["title"] for p in projects]
        assert "Project A" in titles
        assert "Project B" in titles

    def test_list_projects_user_isolation(self, project_manager):
        """User 2 cannot see User 1's projects."""
        project_manager.create_project("user_1", "User1 Project", "")
        project_manager.create_project("user_2", "User2 Project", "")
        user1_projects = project_manager.list_projects("user_1")
        user2_projects = project_manager.list_projects("user_2")
        assert all(p["title"] != "User2 Project" for p in user1_projects)
        assert all(p["title"] != "User1 Project" for p in user2_projects)

    def test_delete_project(self, project_manager):
        """Deleting a project removes it from DB and filesystem."""
        pid = project_manager.create_project("user_1", "To Delete", "")
        path = project_manager.get_project_dir(pid)
        assert path.exists()

        project_manager.delete_project(pid)
        assert not path.exists()
        assert not any(p["id"] == pid for p in project_manager.list_projects("user_1"))

    def test_get_project_purpose(self, project_manager):
        pid = project_manager.create_project("user_1", "Purposeful", "Study CTD anomalies")
        assert project_manager.get_project_purpose(pid) == "Study CTD anomalies"

    def test_get_project_title(self, project_manager):
        pid = project_manager.create_project("user_1", "My Title", "")
        assert project_manager.get_project_title(pid) == "My Title"


class TestChatHistory:
    def test_save_and_load(self, project_manager):
        """Chat messages round-trip through the database."""
        pid = project_manager.create_project("user_1", "Chat Test", "")
        messages = [
            {"role": "user", "content": "What CTD instruments are available?"},
            {"role": "assistant", "content": "Let me search for CTD instruments..."},
        ]
        project_manager.save_chat_history(pid, messages)
        loaded = project_manager.load_chat_history(pid)
        assert len(loaded) == 2
        assert loaded[0]["role"] == "user"
        assert loaded[0]["content"] == "What CTD instruments are available?"
        assert loaded[1]["role"] == "assistant"

    def test_overwrite_history(self, project_manager):
        """Saving new history replaces old history."""
        pid = project_manager.create_project("user_1", "Overwrite Test", "")

        project_manager.save_chat_history(pid, [{"role": "user", "content": "first"}])
        assert len(project_manager.load_chat_history(pid)) == 1

        project_manager.save_chat_history(pid, [
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "response"},
        ])
        loaded = project_manager.load_chat_history(pid)
        assert len(loaded) == 2
        assert loaded[0]["content"] == "second"

    def test_empty_history(self, project_manager):
        """Loading history for a project with no messages returns empty list."""
        pid = project_manager.create_project("user_1", "Empty Chat", "")
        assert project_manager.load_chat_history(pid) == []


class TestFlowchart:
    def test_save_and_load_flowchart(self, project_manager):
        """Flowchart state round-trips through the filesystem."""
        pid = project_manager.create_project("user_1", "Flow Test", "")
        project_manager.save_flowchart(pid, "Download -> Clean -> Plot")
        assert project_manager.load_flowchart(pid) == "Download -> Clean -> Plot"

    def test_empty_flowchart(self, project_manager):
        """No flowchart saved returns empty string."""
        pid = project_manager.create_project("user_1", "No Flow", "")
        assert project_manager.load_flowchart(pid) == ""


class TestPathHelpers:
    def test_path_helpers(self, project_manager):
        """All path helpers return correct subdirectory paths."""
        pid = project_manager.create_project("user_1", "Paths Test", "")
        assert project_manager.get_raw_data_dir(pid).name == "raw_data"
        assert project_manager.get_processed_data_dir(pid).name == "processed_data"
        assert project_manager.get_scripts_dir(pid).name == "scripts"
        assert project_manager.get_plots_dir(pid).name == "plots"
