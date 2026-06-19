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
            logger.info("Connecte a ChromaDB Cloud")
        except Exception as e:
            logger.warning(f"ChromaDB Cloud indisponible ({e}), fallback vers stockage local")
            self._client = chromadb.PersistentClient(
                path="./chroma_data",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                )
            )
            logger.info("Client ChromaDB local initialise")

        return self._client

    def _get_embedding_model(self):
        """Charge le modele d'embedding (lazy loading)."""
        if self._embedding_model is None:
            logger.info(f"Chargement du modele d'embedding: {self.settings.embedding_model}")
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            logger.info("Modele d'embedding charge")
        return self._embedding_model

    def get_or_create_collection(self):
        """Recupere ou cree la collection de documents juridiques."""
        if self._collection is not None:
            return self._collection

        client = self._get_client()
        self._collection = client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"description": "Documents juridiques indexes pour RAG"}
        )
        logger.info(f"Collection '{self.settings.chroma_collection_name}' prete")
        return self._collection

    def add_chunks(self, chunks: List):
        """Ajoute des chunks a la collection avec leurs embeddings.
        
        Args:
            chunks: Liste d'objets Chunk (text, metadata, id)
        """
        if not chunks:
            logger.warning("Aucun chunk a indexer")
            return

        collection = self.get_or_create_collection()
        model = self._get_embedding_model()

        # Taille de batch pour respecter les quotas ChromaDB Cloud
        # Limite gratuite: 300 records par operation
        BATCH_SIZE = 100  # Marge de securite sous les 300
        total_indexed = 0

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            batch = chunks[batch_start:batch_end]

            texts = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            ids = [chunk.id for chunk in batch]

            # Generation des embeddings
            logger.info(f"Generation des embeddings pour {len(batch)} chunks (batch {batch_start//BATCH_SIZE + 1})...")
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            embeddings = embeddings.tolist()

            # Ajout a la collection
            try:
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                total_indexed += len(batch)
                logger.info(f"{len(batch)} chunks indexes (total: {total_indexed}/{len(chunks)})")
            except Exception as e:
                error_msg = str(e)
                if "Quota exceeded" in error_msg or "exceeded quota" in error_msg:
                    logger.error(f"QUOTA CHROMADB EXCEDE: {error_msg}")
                    logger.error("Solutions: 1) Passer a un plan payant sur trychroma.com")
                    logger.error("           2) Utiliser le stockage local (supprimer CHROMA_API_KEY du .env)")
                    logger.error("           3) Demander une augmentation de quota")
                    raise
                else:
                    logger.error(f"Erreur d'ajout a ChromaDB: {e}")
                    raise

        logger.info(f"{total_indexed} chunks indexes dans ChromaDB au total")

    def query(self, query_text: str, n_results: Optional[int] = None) -> Dict:
        """Effectue une recherche semantique sur la collection.
        
        Args:
            query_text: Texte de la requete
            n_results: Nombre de resultats (defaut: config.top_k_retrieval)
            
        Returns:
            Resultats de la recherche avec documents, metadonnees et distances
        """
        collection = self.get_or_create_collection()
        model = self._get_embedding_model()
        
        n_results = n_results or self.settings.top_k_retrieval
        
        # Embedding de la requete
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
            logger.info("Collection videe")
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