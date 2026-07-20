import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import settings

# Use Postgres if provided, otherwise fallback to local SQLite for immediate dev/testing
SQLALCHEMY_DATABASE_URL = getattr(settings, "DATABASE_URL", None)
if not SQLALCHEMY_DATABASE_URL:
    # Default to sqlite in the base directory
    from pathlib import Path
    db_path = Path(__file__).resolve().parent.parent.parent / "aida.db"
    SQLALCHEMY_DATABASE_URL = f"sqlite:///{db_path}"

connect_args = {}
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args=connect_args
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
