"""Engine / session management for the SQLite result store."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base

_ENGINE = None
_SESSION_FACTORY: sessionmaker | None = None


def init_engine(database_path: str | Path, echo: bool = False):
    global _ENGINE, _SESSION_FACTORY
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ENGINE = create_engine(
        f"sqlite:///{path}",
        echo=echo,
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(_ENGINE)
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, future=True)
    return _ENGINE


def get_engine():
    if _ENGINE is None:
        from src.core.config import load_settings

        init_engine(load_settings().paths_resolved.database)
    return _ENGINE


def get_session_factory() -> sessionmaker:
    if _SESSION_FACTORY is None:
        get_engine()
    assert _SESSION_FACTORY is not None
    return _SESSION_FACTORY


@contextmanager
def session_scope() -> Session:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
