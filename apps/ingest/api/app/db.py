from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models.database import Base

settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _ensure_schema_columns() -> None:
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    if "documents" not in table_names:
        return

    columns = {column["name"] for column in inspector.get_columns("documents")}
    if "collection_name" not in columns:
        default_collection = settings.qdrant_collection.replace("'", "''")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE documents "
                    f"ADD COLUMN collection_name VARCHAR(128) NOT NULL DEFAULT '{default_collection}'"
                )
            )
    if "chunking_strategy" not in columns:
        default_strategy = settings.chunking_strategy.replace("'", "''")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE documents "
                    f"ADD COLUMN chunking_strategy VARCHAR(64) NOT NULL DEFAULT '{default_strategy}'"
                )
            )

    indexes = {index["name"] for index in inspector.get_indexes("documents")}
    if "ix_documents_collection_name" not in indexes:
        with engine.begin() as connection:
            connection.execute(text("CREATE INDEX ix_documents_collection_name ON documents (collection_name)"))

    if "chunks" not in table_names:
        return

    chunk_columns = {column["name"] for column in inspector.get_columns("chunks")}
    if "chunking_strategy" not in chunk_columns:
        default_strategy = settings.chunking_strategy.replace("'", "''")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE chunks "
                    f"ADD COLUMN chunking_strategy VARCHAR(64) NOT NULL DEFAULT '{default_strategy}'"
                )
            )


def init_database() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_schema_columns()


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
