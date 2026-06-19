"""Client ChromaDB Cloud pour le stockage des vecteurs."""

import os
import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class ChromaCloudClient:
    """Client pour interagir avec ChromaDB Cloud.

    Utilise le SDK officiel chromadb avec connexion cloud.
    Fallback vers stockage local si les credentials cloud ne sont pas fournis.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._collection = None
        self._embedding_model = None

    def _get_client(self):
        """Initialise et retourne le client ChromaDB."""
        if self._client is not None:
            return self._client

        try:
            # Tentative de connexion ChromaDB Cloud
            self._client = chromadb.CloudClient(
                api_key=self.settings.chroma_api_key,
                tenant=self.settings.chroma_tenant,
                database=self.settings.chroma_database,
            )
            logger.info("✅ Connecté à ChromaDB Cloud")
        except Exception as e:
            logger.warning(f"⚠️ ChromaDB Cloud indisponible ({e}), fallback vers stockage local")
            self._client = chromadb.PersistentClient(
                path="./chroma_data",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                )
            )
            logger.info("✅ Client ChromaDB local initialisé")

        return self._client

    def _get_embedding_model(self):
        """Charge le modèle d'embedding (lazy loading)."""
        if self._embedding_model is None:
            logger.info(f"🔄 Chargement du modèle d'embedding: {self.settings.embedding_model}")
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            logger.info("✅ Modèle d'embedding chargé")
        return self._embedding_model

    def get_or_create_collection(self):
        """Récupère ou crée la collection de documents juridiques."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"description": "Documents juridiques indexés pour RAG"}
        )
        logger.info(f"✅ Collection '{self.settings.chroma_collection_name}' prête")
        return self._collection

    def add_chunks(self, chunks: List):
        """Ajoute des chunks à la collection avec leurs embeddings.

        Args:
            chunks: Liste d'objets Chunk (text, metadata, id)
        """
        if not chunks:
            logger.warning("Aucun chunk à indexer")
            return

        collection = self.get_or_create_collection()
        model = self._get_embedding_model()

        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.id for chunk in chunks]

        # Génération des embeddings
        logger.info(f"🔄 Génération des embeddings pour {len(chunks)} chunks...")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.tolist()

        # Ajout à la collection
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"✅ {len(chunks)} chunks indexés dans ChromaDB")

    def query(self, query_text: str, n_results: Optional[int] = None) -> Dict:
        """Effectue une recherche sémantique sur la collection.

        Args:
            query_text: Texte de la requête
            n_results: Nombre de résultats (défaut: config.top_k_retrieval)

        Returns:
            Résultats de la recherche avec documents, métadonnées et distances
        """
        collection = self.get_or_create_collection()
        model = self._get_embedding_model()

        n_results = n_results or self.settings.top_k_retrieval

        # Embedding de la requête
        query_embedding = model.encode([query_text], convert_to_numpy=True)
        query_embedding = query_embedding.tolist()

        # Recherche
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def count(self) -> int:
        """Retourne le nombre de documents dans la collection."""
        collection = self.get_or_create_collection()
        return collection.count()

    def clear(self):
        """Supprime tous les documents de la collection."""
        client = self._get_client()
        try:
            client.delete_collection(name=self.settings.chroma_collection_name)
            self._collection = None
            logger.info("🗑️ Collection vidée")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {e}")


# Singleton
_chroma_client = None


def get_chroma_client() -> ChromaCloudClient:
    """Retourne une instance singleton du client ChromaDB."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = ChromaCloudClient()
    return _chroma_client
