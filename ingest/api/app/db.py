from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models.database import Base

settings = get_settings()
engine = create_engine(settings.database_url, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _ensure_document_collection_column() -> None:
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
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

    indexes = {index["name"] for index in inspector.get_indexes("documents")}
    if "ix_documents_collection_name" not in indexes:
        with engine.begin() as connection:
            connection.execute(text("CREATE INDEX ix_documents_collection_name ON documents (collection_name)"))


def init_database() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_document_collection_column()


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
