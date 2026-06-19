"""Module de chunking intelligent pour documents juridiques."""

import re
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Chunk:
    """Représente un chunk de texte avec ses métadonnées."""
    text: str
    metadata: Dict
    id: str


class LegalChunker:
    """Chunking spécialisé pour les textes juridiques français.

    Stratégie de découpage :
    1. D'abord par articles (Art. X, Article X)
    2. Puis par alinéas (numérotation 1°, 2°, a), b))
    3. Enfin par paragraphes avec chevauchement
    """

    # Patterns de détection des structures juridiques
    ARTICLE_PATTERN = re.compile(
        r"(?:^|
)\s*(?:Art\.?|Article)\s*(\d+[\w\-]*)\s*[\.\-:]?\s*",
        re.IGNORECASE
    )
    ALINEA_PATTERN = re.compile(
        r"(?:^|
)\s*(\d+[°\.]\s+|\([\d\w]+\)\s+|\d+\)\s+|[a-z]\)\s+)",
        re.IGNORECASE
    )
    SECTION_PATTERN = re.compile(
        r"(?:^|
)\s*(?:Titre|TITRE|Chapitre|CHAPITRE|Section|SECTION)\s+[IVX\d]+",
        re.IGNORECASE
    )

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = 100

    def chunk_document(self, text: str, source: str) -> List[Chunk]:
        """Découpe un document juridique en chunks intelligents.

        Args:
            text: Texte complet du document
            source: Nom du fichier source

        Returns:
            Liste de chunks avec métadonnées enrichies
        """
        # Étape 1 : Découper par articles si possible
        article_splits = self._split_by_articles(text)

        chunks = []
        chunk_idx = 0

        for article_text, article_meta in article_splits:
            # Étape 2 : Découper par alinéas si l'article est trop long
            if len(article_text) > self.chunk_size:
                alinea_splits = self._split_by_alineas(article_text, article_meta)
                for alinea_text, alinea_meta in alinea_splits:
                    if len(alinea_text) > self.chunk_size:
                        # Étape 3 : Découper par paragraphes avec overlap
                        para_chunks = self._split_by_paragraphs(
                            alinea_text, alinea_meta
                        )
                        chunks.extend(para_chunks)
                    else:
                        chunks.append(Chunk(
                            text=alinea_text.strip(),
                            metadata={**alinea_meta, "source": source},
                            id=f"{source}_chunk_{chunk_idx}"
                        ))
                        chunk_idx += 1
            else:
                chunks.append(Chunk(
                    text=article_text.strip(),
                    metadata={**article_meta, "source": source},
                    id=f"{source}_chunk_{chunk_idx}"
                ))
                chunk_idx += 1

        return chunks

    def _split_by_articles(self, text: str) -> List[tuple]:
        """Découpe le texte par articles juridiques."""
        matches = list(self.ARTICLE_PATTERN.finditer(text))

        if len(matches) < 2:
            # Pas assez d'articles, retourner le texte entier
            return [(text, {"type": "section", "article": "global"})]

        splits = []
        for i, match in enumerate(matches):
            start = match.start()
            article_num = match.group(1)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            article_text = text[start:end]

            # Extraire le titre/section parent si présent
            section = self._extract_section_context(text, start)

            splits.append((article_text, {
                "type": "article",
                "article": article_num,
                "section": section,
            }))

        return splits

    def _split_by_alineas(self, text: str, base_meta: dict) -> List[tuple]:
        """Découpe un article par alinéas."""
        matches = list(self.ALINEA_PATTERN.finditer(text))

        if len(matches) < 2:
            return [(text, {**base_meta, "type": "article_complet"})]

        splits = []
        for i, match in enumerate(matches):
            start = match.start()
            alinea_num = match.group(1).strip()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            splits.append((text[start:end], {
                **base_meta,
                "type": "alinea",
                "alinea": alinea_num,
            }))

        return splits

    def _split_by_paragraphs(self, text: str, base_meta: dict) -> List[Chunk]:
        """Découpe par paragraphes avec chevauchement pour les longs textes."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        metadata={**base_meta, "type": "paragraph", "chunk_index": chunk_idx},
                        id=f"{base_meta.get('source', 'doc')}_para_{chunk_idx}"
                    ))
                    chunk_idx += 1

                # Chevauchement : garder les derniers mots du chunk précédent
                if len(current_chunk) > self.chunk_overlap:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                metadata={**base_meta, "type": "paragraph", "chunk_index": chunk_idx},
                id=f"{base_meta.get('source', 'doc')}_para_{chunk_idx}"
            ))

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Extrait le texte de chevauchement depuis la fin du chunk précédent."""
        words = text.split()
        overlap_words = words[-int(self.chunk_overlap / 6):]  # ~6 caractères/mot
        return " ".join(overlap_words)

    def _extract_section_context(self, text: str, position: int) -> str:
        """Extrait le contexte de section (Titre, Chapitre) avant la position donnée."""
        before_text = text[:position]
        matches = list(self.SECTION_PATTERN.finditer(before_text))
        if matches:
            return matches[-1].group(0).strip()
        return ""
