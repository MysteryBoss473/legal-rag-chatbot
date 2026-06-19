"""Application principale FastAPI - Legal RAG Chatbot.

API REST + interface web pour consulter les documents juridiques
via un systeme RAG strict avec l'API Groq et ChromaDB (stockage local).
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

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# === Modeles Pydantic ===


class ChatMessage(BaseModel):
    """Un message dans la conversation."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    """Requete de chat avec historique."""
    message: str = Field(..., min_length=1, max_length=4000)
    history: Optional[List[ChatMessage]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Reponse complete du chatbot."""
    response: str
    sources: List[Dict]


class HealthResponse(BaseModel):
    """Reponse de sante de l'API."""
    status: str
    version: str
    documents_indexed: int


# === Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    logger.info("Demarrage de l'application Legal RAG Chatbot")
    
    # Verification que les documents sont deja indexes
    settings = get_settings()
    data_path = os.path.join(settings.data_dir)
    chroma = get_chroma_client()
    
    try:
        doc_count = chroma.count()
        if doc_count == 0:
            logger.warning("Aucun document indexe. Lancez 'python -m app.indexer' avant de demarrer.")
        else:
            logger.info(f"{doc_count} documents deja indexes et prets")
    except Exception as e:
        logger.warning(f"Impossible de verifier le nombre de documents: {e}")
    
    yield
    
    logger.info("Arret de l'application")


# === Application FastAPI ===

app = FastAPI(
    title="Legal RAG Chatbot",
    description="Chatbot juridique base sur le RAG avec Groq et ChromaDB (stockage local)",
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
    """Verifie l'etat de sante de l'application."""
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
    
    Retourne la reponse token par token pour une experience fluide.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")
    
    # Conversion de l'historique
    history = []
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    async def event_generator():
        """Generateur d'evenements SSE."""
        try:
            for token in rag_engine.generate_response(
                query=request.message,
                conversation_history=history,
            ):
                # Echapper les retours a la ligne pour SSE
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
    """Endpoint de chat sans streaming (reponse complete).
    
    Retourne la reponse complete avec les sources.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message vide")
    
    # Conversion de l'historique
    history = []
    if request.history:
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    try:
        # Generation de la reponse
        response_parts = []
        for token in rag_engine.generate_response(
            query=request.message,
            conversation_history=history,
        ):
            response_parts.append(token)
        
        response_text = "".join(response_parts)
        
        # Recuperation des sources
        sources = rag_engine.get_sources(request.message)
        
        return ChatResponse(
            response=response_text,
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Erreur chat: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de generation: {e}")


@app.get("/api/sources")
async def get_sources(query: str):
    """Recupere les sources pertinentes pour une requete sans generer de reponse.
    
    Args:
        query: Texte de la requete
        
    Returns:
        Liste des sources avec metadonnees
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Requete vide")
    
    try:
        sources = rag_engine.get_sources(query)
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Erreur recuperation sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index")
async def trigger_indexing(clear: bool = False):
    """Declenche manuellement l'indexation des documents PDF.
    
    Args:
        clear: Si True, vide la collection existante avant indexation
        
    Returns:
        Statut de l'indexation
    """
    from app.indexer import index_documents
    try:
        index_documents(clear_existing=clear)
        chroma = get_chroma_client()
        return {
            "status": "success",
            "message": "Indexation terminee",
            "documents_indexed": chroma.count(),
        }
    except Exception as e:
        logger.error(f"Erreur indexation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Gestion des erreurs ===

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'erreurs global."""
    logger.error(f"Erreur non geree: {exc}")
    return HTMLResponse(
        content=f"<h1>Erreur interne</h1><p>{str(exc)}</p>",
        status_code=500,
    )