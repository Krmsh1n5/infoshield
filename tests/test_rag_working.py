"""
tests/test_rag_working.py

Demonstrates that RAGRetriever is working correctly:
  - Index builds successfully from source documents
  - Relevant sources surface for real misinformation claims
  - Unrelated queries score low (no false positives)
  - Retrieved context is usable for LLM injection

Run with:
    pytest tests/test_rag_working.py -v
    pytest tests/test_rag_working.py -v -s        # shows scores
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCES_DIR = PROJECT_ROOT / "pipeline" / "counter_narrative" / "sources"


# ---------------------------------------------------------------------------
# Fixture — single retriever instance shared across all tests in this module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def retriever():
    from pipeline.counter_narrative.rag_retriever import RAGRetriever

    r = RAGRetriever(SOURCES_DIR)
    print(f"\n[RAG] Indexed {r.num_chunks} chunks from {SOURCES_DIR}")
    return r


# ---------------------------------------------------------------------------
# 1. Index health
# ---------------------------------------------------------------------------

class TestIndexHealth:
    """The index must be non-empty and structurally sound."""

    def test_chunks_were_indexed(self, retriever) -> None:
        assert retriever.num_chunks > 0, (
            f"Index is empty — check that {SOURCES_DIR} contains .txt files"
        )

    def test_all_five_source_files_contributed(self, retriever) -> None:
        expected = {
            "health_who", "health_moh", "military_mod",
            "election_cec", "science_covid",
        }
        indexed_sources = set(retriever._sources)
        missing = expected - indexed_sources
        assert not missing, f"These source files produced no chunks: {missing}"

    def test_no_comment_lines_in_chunks(self, retriever) -> None:
        for chunk in retriever._chunks:
            assert not chunk.startswith("#"), (
                f"Comment line leaked into index: {chunk[:60]}"
            )

    def test_no_empty_chunks(self, retriever) -> None:
        for chunk in retriever._chunks:
            assert chunk.strip(), "Empty chunk found in index"


# ---------------------------------------------------------------------------
# 2. Relevance — the right source must appear in top-3 for each claim
# ---------------------------------------------------------------------------

# (query, expected_source, description)
RELEVANCE_CASES = [
    (
        "5G towers are spreading the coronavirus to people",
        "health_who",
        "WHO covers 5G/COVID link directly",
    ),
    (
        "vaccines contain microchips to track the population",
        "health_who",
        "WHO debunks microchip vaccine claims",
    ),
    (
        "Ivermectin cures COVID and the government is hiding it",
        "health_moh",
        "MoH covers national treatment protocols",
    ),
    (
        "mRNA vaccines permanently alter your DNA",
        "science_covid",
        "science_covid covers mRNA mechanism",
    ),
    (
        "ICNIRP radiation levels from antennas cause cancer",
        "science_covid",
        "science_covid covers ICNIRP guidelines",
    ),
    (
        "election results were falsified and ballots were stuffed",
        "election_cec",
        "CEC source covers election integrity",
    ),
    (
        "the army has retreated and official casualty figures are fake",
        "military_mod",
        "MoD source covers official military statements",
    ),
    (
        "bleach injection can cure a viral infection",
        "health_who",
        "WHO/MoH sources cover treatment misinformation",
    ),
]


class TestRelevance:
    """The correct authoritative source must appear in the top-3 results."""

    @pytest.mark.parametrize("query,expected_source,reason", RELEVANCE_CASES)
    def test_correct_source_in_top3(
        self, retriever, query: str, expected_source: str, reason: str
    ) -> None:
        results = retriever.retrieve(query, top_k=3)

        assert results, f"No passages returned for query: '{query}'"

        top_sources = [r.source_file for r in results]
        top_score = results[0].score

        print(
            f"\n  query   : {query[:65]}"
            f"\n  top hits: {top_sources}"
            f"\n  scores  : {[round(r.score, 3) for r in results]}"
        )

        assert expected_source in top_sources, (
            f"RELEVANCE FAILURE — {reason}\n"
            f"  Query    : '{query}'\n"
            f"  Expected : '{expected_source}' in top-3\n"
            f"  Got      : {top_sources}\n"
            f"  Top score: {top_score:.4f}\n\n"
            f"  → The source document may be missing the right vocabulary.\n"
            f"    Add surface forms that match this claim to {expected_source}.txt"
        )

    @pytest.mark.parametrize("query,expected_source,reason", RELEVANCE_CASES)
    def test_top_result_scores_above_noise(
        self, retriever, query: str, expected_source: str, reason: str
    ) -> None:
        results = retriever.retrieve(query, top_k=3)
        assert results, f"No passages returned for query: '{query}'"

        top_score = results[0].score
        assert top_score > 0.05, (
            f"Top score too low ({top_score:.4f}) for query: '{query}'\n"
            f"  The retriever found something but has very low confidence.\n"
            f"  Check that source documents contain terminology matching the claim."
        )


# ---------------------------------------------------------------------------
# 3. Discrimination — unrelated queries must NOT score high
# ---------------------------------------------------------------------------

UNRELATED_QUERIES = [
    "football match result Qarabag FC",
    "best recipe for plov pilaf rice dish",
    "exchange rate USD to AZN today",
    "traffic jam on Nizami street Baku",
    "new smartphone release 2024",
]


class TestDiscrimination:
    """Unrelated queries should not produce confident results."""

    @pytest.mark.parametrize("query", UNRELATED_QUERIES)
    def test_unrelated_query_scores_below_threshold(
        self, retriever, query: str
    ) -> None:
        results = retriever.retrieve(query, top_k=3)

        if not results:
            return  # no results = correct behaviour

        top_score = results[0].score
        print(
            f"\n  unrelated: '{query}'"
            f"\n  top score: {top_score:.4f} from '{results[0].source_file}'"
        )

        assert top_score < 0.20, (
            f"FALSE POSITIVE — unrelated query scored too high\n"
            f"  Query    : '{query}'\n"
            f"  Top score: {top_score:.4f} (threshold: 0.20)\n"
            f"  Source   : '{results[0].source_file}'\n"
            f"  Passage  : '{results[0].text[:100]}'\n\n"
            f"  → Consider adding a min_score floor to retriever.retrieve() calls."
        )


# ---------------------------------------------------------------------------
# 4. Context block — output is fit for LLM injection
# ---------------------------------------------------------------------------

class TestContextBlock:
    """retrieve_as_context() must produce a well-formed prompt block."""

    def test_context_block_is_non_empty_for_relevant_query(
        self, retriever
    ) -> None:
        context, sources = retriever.retrieve_as_context(
            "5G causes COVID-19", top_k=3
        )
        assert context.strip(), "Context block is empty for a clearly relevant query"
        assert sources, "No source names returned alongside context"

    def test_context_block_contains_verified_sources_header(
        self, retriever
    ) -> None:
        context, _ = retriever.retrieve_as_context("vaccine microchip", top_k=2)
        assert "VERIFIED SOURCES" in context, (
            "Context block missing '=== VERIFIED SOURCES ===' header — "
            "LLM system prompt relies on this marker"
        )

    def test_context_block_contains_source_labels(self, retriever) -> None:
        context, sources = retriever.retrieve_as_context(
            "election ballot fraud", top_k=2
        )
        for source in sources:
            assert source in context, (
                f"Source label '{source}' not found in context block.\n"
                f"The LLM cannot attribute facts to a source it cannot see."
            )

    def test_source_names_are_known_files(self, retriever) -> None:
        known = {
            "health_who", "health_moh", "military_mod",
            "election_cec", "science_covid",
        }
        _, sources = retriever.retrieve_as_context("COVID vaccine safety", top_k=5)
        for name in sources:
            assert name in known, (
                f"Unexpected source name '{name}' — "
                f"may indicate a stray .txt file in the sources directory"
            )

    def test_context_block_for_unrelated_query_is_empty_or_low_confidence(
        self, retriever
    ) -> None:
        context, sources = retriever.retrieve_as_context(
            "Qarabag football score", top_k=3, 
        )
        # Either nothing returned, or the scores are low enough that
        # the LLM's grounding instruction ("only use provided sources")
        # will prevent hallucination
        if context.strip():
            results = retriever.retrieve("Qarabag football score", top_k=1)
            assert results[0].score < 0.20, (
                "Unrelated query produced high-confidence context — "
                "this will mislead the LLM grounding step"
            )