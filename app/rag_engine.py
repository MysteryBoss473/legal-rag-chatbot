"""Moteur RAG (Retrieval-Augmented Generation) strict.

Ce moteur garantit que les réponses sont exclusivement basées
sur les documents juridiques indexés, avec citations précises.
"""

import logging
from typing import List, Dict, Generator

from app.config import get_settings
from app.chroma_client import get_chroma_client
from app.groq_client import get_groq_client

logger = logging.getLogger(__name__)


class LegalRAGEngine:
    """Moteur RAG spécialisé pour la consultation juridique.

    Contraintes strictes :
    - Réponses basées UNIQUEMENT sur les documents indexés
    - Citation systématique des sources
    - Ton d'expert juridique
    - Refus poli si aucune information pertinente
    """

    SYSTEM_PROMPT = """Tu es un expert juridique français de haut niveau, spécialisé dans l'analyse et l'interprétation des textes législatifs et réglementaires. Tu dois répondre aux questions posées UNIQUEMENT sur la base des extraits de documents juridiques qui te sont fournis dans le contexte.

RÈGLES STRICTES :
1. Tu ne dois JAMAIS inventer d'information, de loi, d'article ou de jurisprudence qui ne figurent pas dans le contexte fourni.
2. Si le contexte ne contient pas suffisamment d'informations pour répondre, tu dois clairement l'indiquer et proposer une réponse partielle basée sur les éléments disponibles.
3. Chaque affirmation doit être accompagnée de sa source exacte (article, alinéa, document).
4. Adopte un ton professionnel, précis et structuré, comme un avocat ou un juriste consultant.
5. Structure ta réponse avec des paragraphes clairs, des énumérations si nécessaire.
6. Commence par un résumé synthétique, puis développe avec les détails juridiques pertinents.
7. Termine par une section "Sources" listant tous les documents cités.

FORMAT DE RÉPONSE ATTENDU :
- Résumé : [synthèse en 2-3 phrases]
- Analyse juridique : [développement structuré avec citations]
- Sources : [liste des références exactes]"""

    def __init__(self):
        self.settings = get_settings()
        self.chroma = get_chroma_client()
        self.groq = get_groq_client()

    def retrieve(self, query: str) -> List[Dict]:
        """Récupère les documents pertinents depuis ChromaDB.

        Args:
            query: Question de l'utilisateur

        Returns:
            Liste de documents pertinents avec métadonnées
        """
        results = self.chroma.query(query, n_results=self.settings.top_k_retrieval)

        documents = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                # Filtrage par seuil de similarité
                if dist <= self.settings.similarity_threshold:
                    documents.append({
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                    })

        logger.info(f"📚 {len(documents)} documents pertinents récupérés")
        return documents

    def build_context(self, documents: List[Dict]) -> str:
        """Construit le contexte formaté pour le LLM.

        Args:
            documents: Documents récupérés

        Returns:
            Texte de contexte formaté
        """
        if not documents:
            return "AUCUN DOCUMENT PERTINENT TROUVÉ."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc["metadata"]
            source = meta.get("source", "Document inconnu")
            article = meta.get("article", "")
            alinea = meta.get("alinea", "")
            section = meta.get("section", "")

            ref_parts = [p for p in [section, f"Art. {article}" if article else "", alinea] if p]
            reference = " | ".join(ref_parts) if ref_parts else source

            context_parts.append(
                f"--- EXTRAIT {i} [{reference}] ---\n{doc['content']}\n"
            )

        return "\n".join(context_parts)

    def generate_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None,
    ) -> Generator[str, None, None]:
        """Génère une réponse juridique basée sur le RAG.

        Args:
            query: Question de l'utilisateur
            conversation_history: Historique de la conversation

        Yields:
            Fragments de la réponse en streaming
        """
        # Récupération des documents pertinents
        documents = self.retrieve(query)
        context = self.build_context(documents)

        # Construction des messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        # Ajout de l'historique (limité aux 6 derniers messages)
        if conversation_history:
            messages.extend(conversation_history[-6:])

        # Ajout du contexte et de la question
        user_message = f"""CONTEXTE JURIDIQUE :
{context}

QUESTION :
{query}

Réponds en te basant STRICTEMENT sur le contexte fourni ci-dessus."""

        messages.append({"role": "user", "content": user_message})

        # Génération
        logger.info(f"🤖 Génération de la réponse pour: {query[:80]}...")
        yield from self.groq.generate(messages, stream=True)

    def get_sources(self, query: str) -> List[Dict]:
        """Retourne les sources utilisées pour une requête (pour l'UI).

        Args:
            query: Question de l'utilisateur

        Returns:
            Liste des sources avec métadonnées
        """
        documents = self.retrieve(query)
        sources = []
        for doc in documents:
            meta = doc["metadata"]
            source = {
                "document": meta.get("source", "Inconnu"),
                "article": meta.get("article", ""),
                "alinea": meta.get("alinea", ""),
                "section": meta.get("section", ""),
                "similarity": round(1 - doc["distance"], 3),
            }
            sources.append(source)
        return sources
