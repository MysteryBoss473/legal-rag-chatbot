"""Application principale FastAPI - Legal RAG Chatbot.

API REST + interface web pour consulter les documents juridiques
via un système RAG strict avec l'API Groq et ChromaDB Cloud.
"""

import os
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag_engine import LegalRAGEngine
from app.chroma_client import get_chroma_client
from app.indexer import index_documents

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# === Modèles Pydantic ===


class ChatMessage(BaseModel):
    """Un message dans la conversation."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Requête de chat avec historique."""
    message: str = Field(..., min_length=1, max_length=4000)
    history: Optional[List[ChatMessage]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Réponse complète du chatbot."""
    response: str
    sources: List[Dict]


class HealthResponse(BaseModel):
    """Réponse de santé de l'API."""
    status: str
    version: str
    documents_indexed: int


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    logger.info("🚀 Démarrage de l'application Legal RAG Chatbot")

    # Indexation automatique au démarrage si des PDF sont présents
    settings = get_settings()
    data_path = os.path.join(settings.data_dir)
    if os.path.exists(data_path) and any(f.endswith(".pdf") for f in os.listdir(data_path)):
        logger.info("📚 Indexation automatique des documents PDF...")
        try:
            index_documents()
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation: {e}")
    else:
        logger.info("ℹ️ Aucun PDF à indexer au démarrage")

    yield

    logger.info("🛑 Arrêt de l'application")


# === Application FastAPI ===

app = FastAPI(
    title="Legal RAG Chatbot",
    description="Chatbot juridique basé sur le RAG avec Groq et ChromaDB Cloud",
    version="1.0.0",
    lifespan=lifespan,
)

# Templates et fichiers statiques
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Instance du moteur RAG
rag_engine = LegalRAGEngine()


# === Routes ===

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Page d'accueil avec l'interface de chat."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifie l'état de santé de l'application."""
    try:
        chroma = get_chroma_client()
        doc_count = chroma.count()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            documents_indexed=doc_count,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Endpoint de chat avec streaming SSE.

    Retourne la réponse token par token pour une expérience fluide.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")

    # Conversion de l'historique
    history = []
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    async def event_generator():
        """Générateur d'événements SSE."""
        try:
            for token in rag_engine.generate_response(
                query=request.message,
                conversation_history=history,
            ):
                # Échapper les retours à la ligne pour SSE
                safe_token = token.replace("\n", "\\n").replace("\r", "")
                yield f"data: {safe_token}\n\n"

            # Signal de fin
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Erreur streaming: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint de chat sans streaming (réponse complète).

    Retourne la réponse complète avec les sources.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")

    # Conversion de l'historique
    history = []
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    try:
        # Génération de la réponse
        response_parts = []
        for token in rag_engine.generate_response(
            query=request.message,
            conversation_history=history,
        ):
            response_parts.append(token)

        response_text = "".join(response_parts)

        # Récupération des sources
        sources = rag_engine.get_sources(request.message)

        return ChatResponse(
            response=response_text,
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de génération: {e}")


@app.get("/api/sources")
async def get_sources(query: str):
    """Récupère les sources pertinentes pour une requête sans générer de réponse.

    Args:
        query: Texte de la requête

    Returns:
        Liste des sources avec métadonnées
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Requête vide")

    try:
        sources = rag_engine.get_sources(query)
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Erreur récupération sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index")
async def trigger_indexing(clear: bool = False):
    """Déclenche manuellement l'indexation des documents PDF.

    Args:
        clear: Si True, vide la collection existante avant indexation

    Returns:
        Statut de l'indexation
    """
    try:
        index_documents(clear_existing=clear)
        chroma = get_chroma_client()
        return {
            "status": "success",
            "message": "Indexation terminée",
            "documents_indexed": chroma.count(),
        }
    except Exception as e:
        logger.error(f"Erreur indexation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Gestion des erreurs ===

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'erreurs global."""
    logger.error(f"Erreur non gérée: {exc}")
    return HTMLResponse(
        content=f"<h1>Erreur interne</h1><p>{str(exc)}</p>",
        status_code=500,
    )
