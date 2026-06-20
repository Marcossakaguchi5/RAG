import logging
import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.auth import create_access_token, password_is_valid, require_authenticated
from app.db import SessionLocal, get_session, init_database
from app.models import Document
from app.schemas import (
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
from app.services.ingestion import document_to_schema, ingest_pdf
from app.services.indexer import rebuild_index_from_mysql
from app.services.retrieval import retrieve
from app.services.sparse_embeddings import get_sparse_embedding_service
from app.services.storage import ensure_bucket
from app.services.vector_store import ensure_collection, preview_points

logger = logging.getLogger(__name__)
settings = get_settings()


def initialize_dependencies() -> None:
    last_error: Exception | None = None
    for attempt in range(1, 16):
        try:
            init_database()
            ensure_bucket()
            collection_was_rebuilt = ensure_collection()
            get_sparse_embedding_service()
            if collection_was_rebuilt:
                with SessionLocal() as session:
                    reindexed = rebuild_index_from_mysql(session)
                logger.info("Índice Qdrant hybrid reconstruído com %s chunks.", reindexed)
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


@api_router.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: list[UploadFile] = File(...), session: Session = Depends(get_session)
) -> UploadResponse:
    documents: list[DocumentOut] = []
    errors: list[UploadError] = []
    for file in files:
        try:
            documents.append(await ingest_pdf(file, session))
        except ValueError as error:
            errors.append(UploadError(filename=file.filename or "arquivo", detail=str(error)))
        except Exception:
            logger.exception("Falha ao ingerir %s", file.filename)
            errors.append(
                UploadError(filename=file.filename or "arquivo", detail="Não foi possível processar o arquivo.")
            )
        finally:
            await file.close()

    if not documents and errors:
        raise HTTPException(status_code=400, detail=[error.model_dump() for error in errors])
    return UploadResponse(documents=documents, errors=errors)


@api_router.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)) -> list[DocumentOut]:
    documents = list(session.scalars(select(Document).order_by(Document.created_at.desc())))
    return [document_to_schema(document) for document in documents]


@api_router.get("/points", response_model=PointsResponse)
def list_points(limit: int = Query(default=12, ge=1, le=100)) -> PointsResponse:
    try:
        total, points = preview_points(limit)
        previews = [
            PointPreview(
                id=str(point.id),
                document_id=str((point.payload or {}).get("document_id", "")),
                document_name=str((point.payload or {}).get("document_name", "Documento sem nome")),
                page_number=int((point.payload or {}).get("page_number", 0)),
                ordinal=int((point.payload or {}).get("ordinal", 0)),
                content=str((point.payload or {}).get("content", "")),
            )
            for point in points
        ]
        return PointsResponse(total=total, points=previews)
    except Exception as error:
        logger.exception("Falha ao carregar preview de pontos")
        raise HTTPException(status_code=503, detail="Qdrant indisponível.") from error


@api_router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    try:
        results = retrieve(payload.query, payload.method, payload.top_k)
    except Exception as error:
        logger.exception("Falha na recuperação")
        raise HTTPException(status_code=503, detail="A busca está temporariamente indisponível.") from error

    from app.services.evaluation import calculate_metrics

    relevant_ids = list({item.strip() for item in payload.relevant_chunk_ids if item.strip()})
    metrics = calculate_metrics([result.chunk_id for result in results], relevant_ids, payload.top_k)
    return SearchResponse(method=payload.method, top_k=payload.top_k, results=results, metrics=metrics)


app.include_router(api_router)
