import logging
import time

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.auth import create_access_token, password_is_valid, require_authenticated
from app.core.config import get_settings
from app.schemas import CollectionOut, LoginRequest, LoginResponse, RagRequest, RagResponse, RagasReport
from app.services.answering import generate_answer
from app.services.ingest_client import get_ingest_client
from app.services.ragas import evaluate_ragas
from app.services.reranker import rerank

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, version="0.1.0")
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
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Senha invalida.")
    access_token, expires_at = create_access_token()
    return LoginResponse(access_token=access_token, expires_at=expires_at)


@api_router.get("/auth/session")
def auth_session() -> dict[str, str]:
    return {"status": "ok"}


@api_router.get("/collections", response_model=list[CollectionOut])
def list_collections() -> list[CollectionOut]:
    return get_ingest_client().collections()


@api_router.post("/rag", response_model=RagResponse)
def rag(payload: RagRequest) -> RagResponse:
    started = time.perf_counter()
    candidate_k = max(payload.top_k, payload.candidate_k)

    hits = get_ingest_client().search(
        query=payload.query,
        collection_name=payload.collection_name,
        method=payload.method,
        top_k=candidate_k,
    )
    sources = rerank(payload.query, hits, payload.top_k, payload.use_reranker)
    answer = generate_answer(payload.query, sources)

    ragas = RagasReport(evaluated=False, message="A avaliacao RAGAS foi desativada.")
    if payload.evaluate_ragas:
        try:
            ragas = evaluate_ragas(payload.query, answer, sources, payload.reference_answer.strip())
        except Exception as error:
            logger.exception("Falha ao avaliar RAGAS")
            ragas = RagasReport(evaluated=False, message=f"Nao foi possivel calcular RAGAS: {error}")

    return RagResponse(
        answer=answer,
        collection_name=payload.collection_name,
        method=payload.method,
        top_k=payload.top_k,
        candidate_k=candidate_k,
        used_reranker=payload.use_reranker,
        latency_ms=round((time.perf_counter() - started) * 1000),
        sources=sources,
        ragas=ragas,
    )


app.include_router(api_router)
