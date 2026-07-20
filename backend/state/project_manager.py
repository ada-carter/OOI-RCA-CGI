"""
Unified project path and state manager.

Handles project directory creation and resolution.
Now backed by SQLAlchemy for relational persistence of projects and chats.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional

from db.database import SessionLocal
from db.models import Project, ChatMessage

logger = logging.getLogger(__name__)


class ProjectManager:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent / "projects"
        self.current_project_id: Optional[str] = None
        os.makedirs(self.base_dir, exist_ok=True)

    # ── Path Helpers ──

    def get_project_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id

    def get_raw_data_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id / "raw_data"

    def get_processed_data_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id / "processed_data"

    def get_scripts_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id / "scripts"

    def get_plots_dir(self, project_id: str) -> Path:
        return self.base_dir / project_id / "plots"

    # ── Project CRUD ──

    def create_project(self, user_id: str, title: str, purpose: str = "") -> str:
        """Create a new project in the DB and create standard subdirectories."""
        with SessionLocal() as db:
            new_proj = Project(user_id=user_id, title=title, purpose=purpose)
            db.add(new_proj)
            db.commit()
            db.refresh(new_proj)
            project_id = new_proj.id
        
        self.current_project_id = project_id
        project_path = self.base_dir / project_id
        for subdir in ("raw_data", "processed_data", "scripts", "plots", "workflows"):
            os.makedirs(project_path / subdir, exist_ok=True)
            
        logger.info(f"Created project: {project_id} ({title})")
        return project_id

    def delete_project(self, project_id: str):
        """Delete a project from DB and filesystem."""
        with SessionLocal() as db:
            proj = db.query(Project).filter(Project.id == project_id).first()
            if proj:
                db.delete(proj)
                db.commit()

        project_path = self.base_dir / project_id
        if project_path.exists() and project_path.is_dir():
            shutil.rmtree(project_path)
            logger.info(f"Deleted project: {project_id}")
            
        if self.current_project_id == project_id:
            self.current_project_id = None

    def list_projects(self, user_id: str) -> List[dict]:
        """Return list of project dictionaries sorted by creation date."""
        with SessionLocal() as db:
            projects = db.query(Project).filter(Project.user_id == user_id).order_by(Project.created_at.desc()).all()
            return [{"id": p.id, "title": p.title, "purpose": p.purpose} for p in projects]

    def get_project_purpose(self, project_id: str) -> str:
        """Retrieve the project's purpose from DB."""
        with SessionLocal() as db:
            proj = db.query(Project).filter(Project.id == project_id).first()
            return proj.purpose if proj else ""
            
    def get_project_title(self, project_id: str) -> str:
        with SessionLocal() as db:
            proj = db.query(Project).filter(Project.id == project_id).first()
            return proj.title if proj else project_id

    # ── Chat History ──

    def save_chat_history(self, project_id: str, messages: list):
        """Overwrite chat history for the project in DB."""
        with SessionLocal() as db:
            # Delete old messages
            db.query(ChatMessage).filter(ChatMessage.project_id == project_id).delete()
            
            # Insert new messages
            for msg in messages:
                db_msg = ChatMessage(
                    project_id=project_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    meta_data={k: v for k, v in msg.items() if k not in ("role", "content")}
                )
                db.add(db_msg)
            db.commit()

    def load_chat_history(self, project_id: str) -> list:
        with SessionLocal() as db:
            messages = db.query(ChatMessage).filter(ChatMessage.project_id == project_id).order_by(ChatMessage.timestamp).all()
            history = []
            for m in messages:
                msg_dict = {"role": m.role, "content": m.content}
                if m.meta_data:
                    msg_dict.update(m.meta_data)
                history.append(msg_dict)
            return history

    # ── Flowchart State (Still filesystem for now since it's transient UI state) ──

    def save_flowchart(self, project_id: str, steps: str):
        project_path = self.base_dir / project_id
        if not project_path.exists():
            return
        flow_path = project_path / "flowchart_state.txt"
        with open(flow_path, "w", encoding="utf-8") as f:
            f.write(steps)

    def load_flowchart(self, project_id: str) -> str:
        flow_path = self.base_dir / project_id / "flowchart_state.txt"
        if flow_path.exists():
            return flow_path.read_text(encoding="utf-8").strip()
        return ""


project_manager = ProjectManager()
