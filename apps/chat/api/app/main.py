from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.auth import create_access_token, password_is_valid, require_authenticated
from app.core.config import get_settings
from app.schemas import (
    CollectionOut,
    LoginRequest,
    LoginResponse,
    RagRequest,
    RagResponse,
    RagasEvaluationRequest,
    RagasReport,
)
from app.services.ingest_client import get_ingest_client
from app.services.answering import prompt_sha256
from app.services.rag_graph import run_rag_graph, run_ragas_evaluation

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


@api_router.get("/experiment-config")
def experiment_config() -> dict[str, object]:
    """Expose non-secret generation settings used by academic manifests."""

    return {
        "app_version": app.version,
        "generator": {
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        },
        "context": {
            "max_characters": settings.max_context_characters,
            "selection_policy": "rank_order_until_character_limit",
        },
        "prompt": {
            "sha256": prompt_sha256(),
            "language": "pt-BR",
        },
        "reranker": {
            "type": "heuristic_retrieval_score_plus_lexical_overlap",
            "retrieval_weight": 0.62,
            "lexical_weight": 0.38,
        },
        "ragas": {
            "model": settings.resolved_ragas_model,
            "base_url": settings.resolved_ragas_base_url,
            "embedding_model": settings.ragas_embedding_model,
        },
    }


@api_router.post("/rag", response_model=RagResponse)
def rag(payload: RagRequest) -> RagResponse:
    return run_rag_graph(payload)


@api_router.post("/rag/ragas", response_model=RagasReport)
def ragas(payload: RagasEvaluationRequest) -> RagasReport:
    return run_ragas_evaluation(payload)


app.include_router(api_router)
