"""
Optional SQLAlchemy 2.0 ORM database layer for the LS backend.

Key properties:
- No side effects on import. Nothing is initialized until init_db() is called.
- Database backend is selected by environment variables at init time:
  * If USE_POSTGRESQL=true: use POSTGRES_URL (e.g., postgresql+psycopg://user:pass@host/db)
  * Else if SQLITE_PATH is provided: use sqlite:///absolute/path/to/db.sqlite3
  * Else: use in-memory SQLite (sqlite://)

Models:
- InferenceSession
- PredictionRecord

Helpers:
- init_db(): initializes engine, creates tables
- get_session(): returns SQLAlchemy sessionmaker
- save_inference(session_info, predictions): persists a session and its predictions

This module targets Python 3.12 and SQLAlchemy 2.0 style APIs.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

# Module-level handles set by init_db(); kept None until initialization.
_ENGINE: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None


def _resolve_database_url() -> str:
    """Resolve the SQLAlchemy database URL from environment variables.

    Priority:
    1) USE_POSTGRESQL=true -> POSTGRES_URL (required)
    2) SQLITE_PATH -> sqlite:///absolute/path/to/db.sqlite3
    3) Fallback -> in-memory SQLite (sqlite://)
    """
    use_pg = os.environ.get("USE_POSTGRESQL", "").lower() == "true"
    if use_pg:
        pg_url = os.environ.get("POSTGRES_URL")
        if not pg_url:
            raise RuntimeError(
                "USE_POSTGRESQL=true but POSTGRES_URL is not set. Example: postgresql+psycopg://user:pass@host/db"
            )
        return pg_url

    sqlite_path = os.environ.get("SQLITE_PATH")
    if sqlite_path:
        # Ensure absolute path for sqlite file
        abs_path = os.path.abspath(sqlite_path)
        return f"sqlite:///{abs_path}"

    # In-memory SQLite by default
    return "sqlite://"


class Base(DeclarativeBase):
    pass


class InferenceSession(Base):
    __tablename__ = "inference_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    model_version: Mapped[str] = mapped_column(String(length=255), nullable=False)
    source: Mapped[str] = mapped_column(String(length=64), nullable=False)

    predictions: Mapped[List["PredictionRecord"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("inference_sessions.id", ondelete="CASCADE"), nullable=False
    )
    image_path: Mapped[str] = mapped_column(String(length=1024), nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(length=255), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    raw_response: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)

    session: Mapped[InferenceSession] = relationship(back_populates="predictions")


def init_db(echo: bool = False) -> Engine:
    """Initialize the database engine, create all tables, and prepare a session factory.

    This function has side effects only when called; importing the module does nothing.
    """
    global _ENGINE, _SessionLocal

    url = _resolve_database_url()
    engine = create_engine(url, echo=echo, pool_pre_ping=True, future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(engine, class_=Session, expire_on_commit=False, future=True)

    _ENGINE = engine
    _SessionLocal = SessionLocal
    return engine


def get_session() -> sessionmaker[Session]:
    """Return the configured sessionmaker.

    init_db() must be called first; otherwise raises RuntimeError.
    """
    if _SessionLocal is None:
        raise RuntimeError("Database has not been initialized. Call init_db() first.")
    return _SessionLocal


def save_inference(session_info: Dict[str, Any], predictions: Iterable[Dict[str, Any]]) -> int:
    """Persist an inference session and its prediction records.

    Parameters:
        session_info: Dict with keys:
            - timestamp: datetime (optional; defaults to now if missing)
            - model_version: str (required)
            - source: str (required; e.g., 'API', 'batch', 'LS')
        predictions: Iterable of dicts, each with keys:
            - image_path: str
            - x, y, width, height: float
            - label: str
            - score: float
            - raw_response: JSON-serializable dict (optional)

    Returns:
        The created InferenceSession.id
    """
    SessionLocal = get_session()
    ts: datetime = session_info.get("timestamp") or datetime.utcnow()
    model_version: str = str(session_info.get("model_version", ""))
    source: str = str(session_info.get("source", ""))

    if not model_version or not source:
        raise ValueError("session_info must include 'model_version' and 'source'")

    sess = InferenceSession(timestamp=ts, model_version=model_version, source=source)

    for p in predictions:
        # Defensive parsing with clear errors if essential fields are missing
        try:
            record = PredictionRecord(
                image_path=str(p["image_path"]),
                x=float(p["x"]),
                y=float(p["y"]),
                width=float(p["width"]),
                height=float(p["height"]),
                label=str(p["label"]),
                score=float(p["score"]),
                raw_response=p.get("raw_response"),
            )
        except KeyError as e:
            raise ValueError(f"Prediction missing required field: {e}") from e
        sess.predictions.append(record)

    with SessionLocal() as db:  # type: ignore[operator]
        db.add(sess)
        db.commit()
        db.refresh(sess)
        return int(sess.id)
