"""Moteur RAG (Retrieval-Augmented Generation) strict.

Ce moteur garantit que les reponses sont exclusivement basees
sur les documents juridiques indexes, avec citations precises.
"""

import logging
from typing import List, Dict, Generator

from app.config import get_settings
from app.chroma_client import get_chroma_client
from app.groq_client import get_groq_client

logger = logging.getLogger(__name__)


class LegalRAGEngine:
    """Moteur RAG specialise pour la consultation juridique.
    
    Contraintes strictes :
    - Reponses basees UNIQUEMENT sur les documents indexes
    - Citation systematique des sources
    - Ton d'expert juridique
    - Refus poli si aucune information pertinente
    """

    SYSTEM_PROMPT = """Tu es un expert juridique francais de haut niveau, specialise dans l'analyse et l'interpretation des textes legislatifs et reglementaires. Tu dois repondre aux questions posees UNIQUEMENT sur la base des extraits de documents juridiques qui te sont fournis dans le contexte.

REGLES STRICTES :
1. Tu ne dois JAMAIS inventer d'information, de loi, d'article ou de jurisprudence qui ne figurent pas dans le contexte fourni.
2. Si le contexte ne contient pas suffisamment d'informations pour repondre, tu dois clairement l'indiquer et proposer une reponse partielle basee sur les elements disponibles.
3. Chaque affirmation doit etre accompagnee de sa source exacte avec le NOM COMPLET DU DOCUMENT (ex: "Code de la route, Article 12" ou "Loi N°001-2010, Article 5").
4. Adopte un ton professionnel, precis et structure, comme un avocat ou un juriste consultant.
5. Structure ta reponse avec des paragraphes clairs, des enumerations si necessaire.
6. Commence par un resume synthetique, puis developpe avec les details juridiques pertinents.
7. Termine par une section "Sources" listant tous les documents cites avec leur nom complet.

FORMAT DE REPONSE ATTENDU :
- Resume : [synthese en 2-3 phrases]
- Analyse juridique : [developpement structure avec citations incluant le nom du document]
- Sources : [liste des references exactes avec nom du document + article]"""

    def __init__(self):
        self.settings = get_settings()
        self.chroma = get_chroma_client()
        self.groq = get_groq_client()

    def retrieve(self, query: str) -> List[Dict]:
        """Recupere les documents pertinents depuis ChromaDB.
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Liste de documents pertinents avec metadonnees
        """
        # Recuperer plus de resultats pour avoir le choix, mais on filtrera apres
        n_results = 15
        results = self.chroma.query(query, n_results=n_results)
        
        documents = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []
            
            logger.info(f"Retrieved {len(docs)} raw results from ChromaDB")
            
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
                similarity_score = max(0, 1 - (dist ** 2) / 4.0)
                
                logger.debug(f"Result {i}: distance={dist:.4f}, approx_cos_sim={similarity_score:.4f}")
                
                # Seuil tres permissif pour ne rien manquer
                if dist < 2.5:
                    documents.append({
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                        "similarity": round(similarity_score, 3),
                    })
                else:
                    logger.debug(f"Filtered out result {i} (distance={dist:.4f} >= 2.5)")
        
        logger.info(f"{len(documents)} documents pertinents retenus apres filtrage")
        return documents

    def build_context(self, documents: List[Dict]) -> str:
        """Construit le contexte formate pour le LLM.
        
        Limite le contexte pour ne pas depasser la limite de tokens Groq.
        Environ 800 tokens = ~3200 caracteres pour le contexte.
        
        Args:
            documents: Documents recuperes
            
        Returns:
            Texte de contexte formate
        """
        if not documents:
            return "AUCUN DOCUMENT PERTINENT TROUVE."
        
        # Limiter le nombre de chunks dans le contexte pour respecter la limite TPM
        MAX_CONTEXT_CHARS = 3000  # ~750 tokens
        MAX_CHUNKS = 5
        
        context_parts = []
        total_chars = 0
        
        for i, doc in enumerate(documents[:MAX_CHUNKS], 1):
            meta = doc["metadata"]
            source = meta.get("source", "Document inconnu")
            doc_type = meta.get("document_type", "")
            article = meta.get("article", "")
            alinea = meta.get("alinea", "")
            section = meta.get("section", "")
            
            # Construire la reference complete
            ref_parts = []
            if doc_type:
                ref_parts.append(doc_type)
            ref_parts.append(source)
            if section:
                ref_parts.append(section)
            if article and article != "global":
                ref_parts.append(f"Art. {article}")
            if alinea:
                ref_parts.append(alinea)
            
            reference = " | ".join(ref_parts)
            
            # Tronquer le contenu si necessaire pour rester dans la limite
            content = doc['content']
            max_content_len = max(200, (MAX_CONTEXT_CHARS - total_chars) // (MAX_CHUNKS - i + 1) - 100)
            if len(content) > max_content_len:
                content = content[:max_content_len] + "..."
            
            chunk_text = f"--- EXTRAIT {i} [{reference}] ---\n{content}\n"
            total_chars += len(chunk_text)
            context_parts.append(chunk_text)
            
            if total_chars >= MAX_CONTEXT_CHARS:
                break
        
        return "\n".join(context_parts)

    def generate_response(
        self,
        query: str,
        conversation_history: List[Dict[str, str]] = None,
    ) -> Generator[str, None, None]:
        """Genere une reponse juridique basee sur le RAG.
        
        Args:
            query: Question de l'utilisateur
            conversation_history: Historique de la conversation
            
        Yields:
            Fragments de la reponse en streaming
        """
        # Recuperation des documents pertinents
        documents = self.retrieve(query)
        context = self.build_context(documents)
        
        # Construction des messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]
        
        # Ajout de l'historique (limite aux 4 derniers messages pour economiser les tokens)
        if conversation_history:
            messages.extend(conversation_history[-4:])
        
        # Ajout du contexte et de la question
        user_message = f"""CONTEXTE JURIDIQUE :
{context}

QUESTION :
{query}

Reponds en te basant STRICTEMENT sur le contexte fourni ci-dessus.
N'oublie pas de citer le NOM COMPLET DU DOCUMENT pour chaque affirmation."""
        
        messages.append({"role": "user", "content": user_message})
        
        # Log de la taille approximative
        total_chars = sum(len(m["content"]) for m in messages)
        logger.info(f"Taille approximative du prompt: {total_chars} chars (~{total_chars//4} tokens)")
        
        # Generation
        logger.info(f"Generation de la reponse pour: {query[:80]}...")
        yield from self.groq.generate(messages, stream=True)

    def get_sources(self, query: str) -> List[Dict]:
        """Retourne les sources utilisees pour une requete (pour l'UI).
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Liste des sources avec metadonnees
        """
        documents = self.retrieve(query)
        sources = []
        for doc in documents:
            meta = doc["metadata"]
            source = {
                "document": meta.get("source", "Inconnu"),
                "document_type": meta.get("document_type", ""),
                "article": meta.get("article", ""),
                "alinea": meta.get("alinea", ""),
                "section": meta.get("section", ""),
                "similarity": doc.get("similarity", 0),
            }
            sources.append(source)
        return sources