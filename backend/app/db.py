"""
SQLAlchemy engine/session setup and the declarative Base.

SQLite is used per the project plan (cheap, zero-config, file-based),
which is perfectly adequate for scan history. Swap DATABASE_URL to a
hosted Postgres later without touching model code (SQLAlchemy dialects).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

# check_same_thread=False: FastAPI may use the connection from different
# threads; SQLite default forbids cross-thread reuse, so relax it here.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)

# expire_on_commit=False: objects stay readable after commit (needed when we
# build the API response right after inserting a scan).
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

Base = declarative_base()


def init_db() -> None:
    """Create all tables that don't exist yet. Called once at startup."""
    from . import models  # noqa: F401  (import registers the table classes)

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yield a session per request, always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
