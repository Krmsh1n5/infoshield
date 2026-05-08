"""
pipeline/counter_narrative/rag_retriever.py

In-memory TF-IDF retriever over local verified source documents.
No external vector DB — intentionally lightweight for deployment in
constrained / air-gapped government environments.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RetrievedPassage(NamedTuple):
    """A single retrieved passage with provenance."""
    text: str
    source_file: str        # stem of the .txt file, e.g. "health_who"
    score: float            # cosine similarity ∈ [0, 1]


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _chunk_document(text: str, chunk_size: int = 3) -> list[str]:
    """
    Split a document into overlapping sentence windows.

    Parameters
    ----------
    text       : raw document text (may contain comment lines starting with #)
    chunk_size : number of sentences per chunk

    Returns
    -------
    List of string chunks (non-empty lines only; comment lines stripped).
    """
    # Strip markdown-style comment lines and blank lines
    clean_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    combined = " ".join(clean_lines)

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(combined) if s.strip()]
    if not sentences:
        return [combined] if combined else []

    chunks: list[str] = []
    step = max(1, chunk_size - 1)           # one-sentence overlap
    for i in range(0, len(sentences), step):
        window = sentences[i : i + chunk_size]
        if window:
            chunks.append(" ".join(window))

    return chunks


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    TF-IDF retriever over a directory of .txt source documents.

    Usage
    -----
    retriever = RAGRetriever(Path("pipeline/counter_narrative/sources"))
    passages  = retriever.retrieve("5G towers spread COVID-19", top_k=3)
    for p in passages:
        print(p.source_file, p.score, p.text[:80])
    """

    def __init__(
        self,
        sources_dir: Path,
        chunk_size: int = 3,
        min_chunk_length: int = 20,
    ) -> None:
        """
        Parameters
        ----------
        sources_dir       : directory containing .txt verified source files
        chunk_size        : sentences per TF-IDF chunk
        min_chunk_length  : discard chunks shorter than this many characters
        """
        self.sources_dir = Path(sources_dir)
        self.chunk_size = chunk_size
        self.min_chunk_length = min_chunk_length

        self._chunks: list[str] = []            # parallel with _sources
        self._sources: list[str] = []           # stem names, e.g. "health_who"
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None                     # sparse TF-IDF matrix

        self._load_and_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_index(self) -> None:
        """Load all .txt files and build TF-IDF index."""
        txt_files = sorted(self.sources_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(
                "RAGRetriever: no .txt files found in %s", self.sources_dir
            )

        for path in txt_files:
            try:
                raw = path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to read %s: %s", path, exc)
                continue

            doc_chunks = _chunk_document(raw, self.chunk_size)
            for chunk in doc_chunks:
                if len(chunk) >= self.min_chunk_length:
                    self._chunks.append(chunk)
                    self._sources.append(path.stem)

        if not self._chunks:
            logger.warning("RAGRetriever: index is empty — no chunks loaded.")
            return

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._matrix = self._vectorizer.fit_transform(self._chunks)
        logger.info(
            "RAGRetriever: indexed %d chunks from %d files.",
            len(self._chunks),
            len(txt_files),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.01,
    ) -> list[RetrievedPassage]:
        """
        Return the top_k most relevant passages for *query*.

        Parameters
        ----------
        query     : the misinformation claim or topic string
        top_k     : maximum number of passages to return
        min_score : discard passages below this cosine similarity

        Returns
        -------
        List of RetrievedPassage sorted by descending score.
        Returns [] if the index is empty or no passage meets min_score.
        """
        if self._vectorizer is None or self._matrix is None:
            logger.warning("RAGRetriever.retrieve called on empty index.")
            return []

        query_vec = self._vectorizer.transform([query])
        scores: np.ndarray = cosine_similarity(query_vec, self._matrix).flatten()

        # Rank and filter
        ranked_idx = np.argsort(scores)[::-1]
        results: list[RetrievedPassage] = []
        seen_sources: set[str] = set()

        for idx in ranked_idx:
            if len(results) >= top_k:
                break
            score = float(scores[idx])
            if score < min_score:
                break
            results.append(
                RetrievedPassage(
                    text=self._chunks[idx],
                    source_file=self._sources[idx],
                    score=score,
                )
            )
            seen_sources.add(self._sources[idx])

        logger.debug(
            "RAGRetriever: query=%r → %d passages (sources: %s)",
            query[:60],
            len(results),
            sorted(seen_sources),
        )
        return results

    def retrieve_as_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> tuple[str, list[str]]:
        """
        Convenience wrapper that returns:
          - a formatted context block ready for LLM injection
          - a list of source_file names that were cited

        Returns
        -------
        (context_text, source_names)
        """
        passages = self.retrieve(query, top_k=top_k)
        if not passages:
            return "", []

        lines: list[str] = ["=== VERIFIED SOURCES ==="]
        for i, p in enumerate(passages, 1):
            lines.append(f"[Source {i} — {p.source_file}]")
            lines.append(p.text)
            lines.append("")

        return "\n".join(lines), [p.source_file for p in passages]
