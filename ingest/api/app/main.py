import logging
import time
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.core.auth import create_access_token, password_is_valid, require_authenticated
from app.db import SessionLocal, get_session, init_database
from app.models import Document, KnowledgeCollection
from app.schemas import (
    CollectionIn,
    CollectionOut,
    DocumentOut,
    LoginRequest,
    LoginResponse,
    PointPreview,
    PointsResponse,
    SearchRequest,
    SearchResponse,
    UploadError,
    UploadResponse,
)
from app.services.collections import normalize_collection_name
from app.services.ingestion import document_to_schema, ingest_pdf
from app.services.indexer import rebuild_index_from_mysql
from app.services.pdf_processor import CHUNKING_STRATEGIES, normalize_chunking_strategy
from app.services.retrieval import retrieve
from app.services.sparse_embeddings import get_sparse_embedding_service
from app.services.storage import ensure_bucket
from app.services.vector_store import ensure_collection, preview_points

logger = logging.getLogger(__name__)
ingestion_lock = Lock()
settings = get_settings()


def ensure_collection_ready(collection_name: str, session: Session | None = None) -> str:
    collection_name = normalize_collection_name(collection_name)
    collection_was_rebuilt = ensure_collection(collection_name)
    if collection_was_rebuilt and session is not None:
        reindexed = rebuild_index_from_mysql(session, collection_name=collection_name)
        logger.info("Collection %s reconstruída com %s chunks.", collection_name, reindexed)
    return collection_name


def ensure_collection_record(session: Session, collection_name: str) -> None:
    exists = session.scalar(select(KnowledgeCollection.id).where(KnowledgeCollection.name == collection_name))
    if exists is None:
        session.add(KnowledgeCollection(name=collection_name))
        session.commit()


def initialize_dependencies() -> None:
    last_error: Exception | None = None
    for attempt in range(1, 16):
        try:
            init_database()
            ensure_bucket()
            get_sparse_embedding_service()
            with SessionLocal() as session:
                collection_names = set(
                    session.scalars(select(Document.collection_name).where(Document.collection_name != "").distinct())
                )
                collection_names.update(session.scalars(select(KnowledgeCollection.name)))
                collection_names.add(settings.qdrant_collection)
                for collection_name in sorted(collection_names):
                    ensure_collection_ready(collection_name, session)
            return
        except Exception as error:  # serviços podem ainda estar subindo no compose
            last_error = error
            logger.warning("Dependências indisponíveis (%s/15): %s", attempt, error)
            time.sleep(2)
    raise RuntimeError("Não foi possível inicializar as dependências da API.") from last_error


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_dependencies()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api_router = APIRouter(prefix="/api", dependencies=[Depends(require_authenticated)])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    if not password_is_valid(payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Senha inválida.")
    access_token, expires_at = create_access_token()
    return LoginResponse(access_token=access_token, expires_at=expires_at)


@api_router.get("/auth/session")
def auth_session() -> dict[str, str]:
    return {"status": "ok"}


@api_router.get("/collections", response_model=list[CollectionOut])
def list_collections(session: Session = Depends(get_session)) -> list[CollectionOut]:
    rows = session.execute(
        select(Document.collection_name, func.count(Document.id))
        .group_by(Document.collection_name)
        .order_by(Document.collection_name)
    ).all()
    counts = {name: count for name, count in rows if name}
    names = set(counts)
    names.update(session.scalars(select(KnowledgeCollection.name)))
    names.add(settings.qdrant_collection)
    return [CollectionOut(name=name, documents_count=counts.get(name, 0)) for name in sorted(names)]


@api_router.post("/collections", response_model=CollectionOut, status_code=status.HTTP_201_CREATED)
def create_collection(payload: CollectionIn, session: Session = Depends(get_session)) -> CollectionOut:
    try:
        collection_name = ensure_collection_ready(payload.name, session)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    ensure_collection_record(session, collection_name)
    documents_count = session.scalar(
        select(func.count(Document.id)).where(Document.collection_name == collection_name)
    ) or 0
    return CollectionOut(name=collection_name, documents_count=documents_count)


@api_router.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_documents(
    files: list[UploadFile] = File(...),
    collection_name: str = Form(default=settings.qdrant_collection),
    chunking_strategy: str = Form(default=settings.chunking_strategy),
    session: Session = Depends(get_session),
) -> UploadResponse:
    if not ingestion_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Já existe uma ingestão em andamento. Tente novamente quando ela terminar.",
        )

    try:
        try:
            collection_name = ensure_collection_ready(collection_name, session)
            chunking_strategy = normalize_chunking_strategy(chunking_strategy)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        ensure_collection_record(session, collection_name)

        documents: list[DocumentOut] = []
        errors: list[UploadError] = []
        for file in files:
            try:
                documents.append(ingest_pdf(file, session, collection_name, chunking_strategy))
            except ValueError as error:
                errors.append(UploadError(filename=file.filename or "arquivo", detail=str(error)))
            except Exception:
                logger.exception("Falha ao ingerir %s", file.filename)
                errors.append(
                    UploadError(filename=file.filename or "arquivo", detail="Não foi possível processar o arquivo.")
                )
            finally:
                file.file.close()

        if not documents and errors:
            raise HTTPException(status_code=400, detail=[error.model_dump() for error in errors])
        return UploadResponse(documents=documents, errors=errors)
    finally:
        ingestion_lock.release()


@api_router.get("/documents", response_model=list[DocumentOut])
def list_documents(
    collection_name: str = Query(default=settings.qdrant_collection),
    session: Session = Depends(get_session),
) -> list[DocumentOut]:
    try:
        collection_name = normalize_collection_name(collection_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    documents = list(
        session.scalars(
            select(Document)
            .where(Document.collection_name == collection_name)
            .options(selectinload(Document.chunks))
            .order_by(Document.created_at.desc())
        )
    )
    return [document_to_schema(document) for document in documents]


@api_router.get("/points", response_model=PointsResponse)
def list_points(
    limit: int = Query(default=12, ge=1, le=100),
    collection_name: str = Query(default=settings.qdrant_collection),
    session: Session = Depends(get_session),
) -> PointsResponse:
    try:
        collection_name = ensure_collection_ready(collection_name, session)
        total, points = preview_points(limit, collection_name)
        previews = [
            PointPreview(
                id=str(point.id),
                document_id=str((point.payload or {}).get("document_id", "")),
                document_name=str((point.payload or {}).get("document_name", "Documento sem nome")),
                file_name=str((point.payload or {}).get("file_name", "")),
                page_number=int((point.payload or {}).get("page_number", 0)),
                ordinal=int((point.payload or {}).get("ordinal", 0)),
                chunk_index=int((point.payload or {}).get("chunk_index", 0)),
                chunk_total=int((point.payload or {}).get("chunk_total", 0)),
                word_count=int((point.payload or {}).get("word_count", 0)),
                char_count=int((point.payload or {}).get("char_count", 0)),
                chunking_strategy=str((point.payload or {}).get("chunking_strategy", settings.chunking_strategy)),
                content=str((point.payload or {}).get("content", "")),
            )
            for point in points
        ]
        return PointsResponse(total=total, points=previews)
    except Exception as error:
        logger.exception("Falha ao carregar preview de pontos")
        raise HTTPException(status_code=503, detail="Qdrant indisponível.") from error


@api_router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, session: Session = Depends(get_session)) -> SearchResponse:
    try:
        collection_name = ensure_collection_ready(payload.collection_name, session)
        results = retrieve(payload.query, payload.method, payload.top_k, collection_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Falha na recuperação")
        raise HTTPException(status_code=503, detail="A busca está temporariamente indisponível.") from error

    from app.services.evaluation import calculate_metrics

    relevant_ids = list({item.strip() for item in payload.relevant_chunk_ids if item.strip()})
    metrics = calculate_metrics([result.chunk_id for result in results], relevant_ids, payload.top_k)
    return SearchResponse(method=payload.method, top_k=payload.top_k, results=results, metrics=metrics)


@api_router.get("/chunking-strategies")
def list_chunking_strategies() -> dict[str, object]:
    return {
        "default": normalize_chunking_strategy(settings.chunking_strategy),
        "strategies": [
            {"value": value, "label": label}
            for value, label in CHUNKING_STRATEGIES.items()
        ],
    }


app.include_router(api_router)
