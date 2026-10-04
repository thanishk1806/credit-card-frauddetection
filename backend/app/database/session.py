"""
SQLAlchemy engine / session setup for SQLite / PostgreSQL with automatic column migration.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables and auto-migrate missing columns on existing SQLite/PostgreSQL databases."""
    from app.models import prediction, user  # noqa: F401  (register models)

    Base.metadata.create_all(bind=engine)

    # Automatically migrate missing columns if table already existed prior to schema upgrade
    try:
        with engine.begin() as conn:
            if settings.DATABASE_URL.startswith("sqlite"):
                result = conn.execute(text("PRAGMA table_info(predictions)"))
                existing_cols = {row[1] for row in result.fetchall()}
                new_cols = [
                    ("amount", "FLOAT"),
                    ("transaction_type", "VARCHAR(32)"),
                    ("merchant_category", "VARCHAR(32)"),
                    ("location", "VARCHAR(128)"),
                    ("device_type", "VARCHAR(32)"),
                    ("card_present", "BOOLEAN DEFAULT 0"),
                    ("international_transaction", "BOOLEAN DEFAULT 0"),
                    ("transaction_timestamp", "DATETIME"),
                    ("derived_features", "JSON"),
                ]
                for col_name, col_type in new_cols:
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}"))
    except Exception:
        pass
