from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.auth import create_access_token, password_is_valid, require_authenticated
from app.core.config import get_settings
from app.schemas import CollectionOut, LoginRequest, LoginResponse, RagRequest, RagResponse
from app.services.ingest_client import get_ingest_client
from app.services.rag_graph import run_rag_graph

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
    return run_rag_graph(payload)


app.include_router(api_router)
