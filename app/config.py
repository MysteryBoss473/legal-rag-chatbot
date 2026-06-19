"""Configuration centralisee de l'application."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Parametres de l'application charges depuis les variables d'environnement."""

    # === Groq ===
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 4096
    groq_timeout: float = 60.0

    # === ChromaDB Cloud (optionnel, fallback local si vide) ===
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    chroma_collection_name: str = "legal_documents"

    # === RAG ===
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 10  # Augmente pour plus de resultats
    similarity_threshold: float = 2.5  # Distance L2 max (tres permissif)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # === App ===
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    # === Paths ===
    data_dir: str = "scrapping/data"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Retourne une instance singleton des parametres."""
    return Settings()