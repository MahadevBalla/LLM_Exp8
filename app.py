import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from support_rag import answer_question, initialize_rag, is_ready

# Basic logs are useful for debugging, but don't log question text or secrets.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("experiment7")

STATIC_DIR = Path(__file__).resolve().parent / "static"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

_hits = defaultdict(deque)
_hits_lock = threading.Lock()


def _warm_up():
    """Build the index in the background so the server starts listening immediately."""
    try:
        initialize_rag()
        logger.info("RAG components initialized")
    except Exception:
        # Details stay in the server log. The next /ask call retries initialization.
        logger.exception("RAG initialization failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(
    title="Experiment 7 - Production-Oriented RAG API",
    description="Clinic FAQ RAG API with validation, logging, rate limiting and latency measurement.",
    version="2.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to answer using the supplied knowledge base (maximum 1000 characters).",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank.")
        return value.strip()


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    response_time_ms: float


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _within_rate_limit(ip: str) -> bool:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True
    now = time.monotonic()
    with _hits_lock:
        window = _hits[ip]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= RATE_LIMIT_PER_MINUTE:
            return False
        window.append(now)
        return True


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    # Confirms the API process is responding. It does not prove that Groq or
    # Hugging Face are reachable; knowledge_base_ready only says the index is built.
    return {
        "status": "ok",
        "service": "experiment7-rag-api",
        "knowledge_base_ready": is_ready(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    if not _within_rate_limit(_client_ip(request)):
        logger.warning("ask_rate_limited")
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again.",
        )

    started = time.perf_counter()
    try:
        result = answer_question(payload.question)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("ask_success latency_ms=%.2f source_count=%d",
                    elapsed_ms, len(result["sources"]))
        return AskResponse(
            answer=result["answer"],
            sources=result["sources"],
            response_time_ms=elapsed_ms,
        )
    except Exception:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.exception("ask_failed latency_ms=%.2f", elapsed_ms)
        # Never send stack traces, provider URLs, or credentials to clients.
        raise HTTPException(
            status_code=502,
            detail="The answer service is temporarily unavailable. Please try again later.",
        )
