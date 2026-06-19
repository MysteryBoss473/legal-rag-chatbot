"""Module d'indexation des documents PDF juridiques.

Extrait le texte des PDF, effectue un chunking intelligent
et indexe les vecteurs dans ChromaDB Cloud.
"""

import os
import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.config import get_settings
from app.chunker import LegalChunker
from app.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrait le texte brut d'un fichier PDF.

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        Texte extrait
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                text += f"\n\n--- Page {page_num} ---\n{page_text}"
        logger.info(f"📄 {pdf_path}: {len(text)} caractères extraits")
    except Exception as e:
        logger.error(f"❌ Erreur extraction {pdf_path}: {e}")

    return text


def index_documents(data_dir: str = None, clear_existing: bool = False):
    """Indexe tous les documents PDF du dossier data.

    Args:
        data_dir: Dossier contenant les PDF (défaut: config.data_dir)
        clear_existing: Si True, vide la collection existante avant indexation
    """
    settings = get_settings()
    data_dir = data_dir or settings.data_dir

    # Vérification du dossier
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"⚠️ Dossier {data_dir} introuvable. Création...")
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Dossier {data_dir} créé. Placez vos PDF ici.")
        return

    # Liste des PDF
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️ Aucun PDF trouvé dans {data_dir}")
        return

    logger.info(f"📚 {len(pdf_files)} PDF trouvés dans {data_dir}")

    # Initialisation
    chroma = get_chroma_client()
    chunker = LegalChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Optionnel : vider la collection
    if clear_existing:
        logger.info("🗑️ Vidage de la collection existante...")
        chroma.clear()

    # Indexation
    total_chunks = 0
    for pdf_file in pdf_files:
        logger.info(f"\n🔍 Indexation: {pdf_file.name}")

        # Extraction
        text = extract_text_from_pdf(str(pdf_file))
        if not text.strip():
            logger.warning(f"⚠️ {pdf_file.name}: texte vide, ignoré")
            continue

        # Chunking
        chunks = chunker.chunk_document(text, source=pdf_file.name)
        logger.info(f"   → {len(chunks)} chunks générés")

        # Indexation
        chroma.add_chunks(chunks)
        total_chunks += len(chunks)

    logger.info(f"\n✅ Indexation terminée: {total_chunks} chunks indexés au total")
    logger.info(f"📊 Documents dans la collection: {chroma.count()}")


def main():
    """Point d'entrée pour l'indexation en ligne de commande."""
    import sys

    clear_flag = "--clear" in sys.argv
    index_documents(clear_existing=clear_flag)


if __name__ == "__main__":
    main()
