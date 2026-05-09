"""
pipeline/counter_narrative
==========================

RAG-grounded counter-narrative generation for InfoShield.

Produces short, factual rebuttals in Azerbaijani (az), Russian (ru),
or English (en) for misinformation claims detected by the BiGCN/SBM
pipeline, using verified governmental and scientific source documents
as grounding context.

Quick start
-----------
    from pipeline.counter_narrative import (
        CounterNarrativeGenerator,
        PostFormatter,
        RebuttalResult,
    )

    gen = CounterNarrativeGenerator()          # uses ANTHROPIC_API_KEY env var
    result = gen.generate_rebuttal(
        false_claim="5G towers spread COVID-19",
        topic="health",
        language="az",
        confidence=0.92,
        cascade_pattern="wide_burst",
    )

    formatter = PostFormatter()
    telegram_post = formatter.format_for_platform(result, "telegram")
    print(telegram_post)

Integration points
------------------
- pipeline/ingestors/whatsapp_bot.py  — after graph reconstruction
- api/server.py                       — /rebuttal endpoint
- dashboard/index.html                — via API call
"""

from .formatter import FormattedPost, PostFormatter
from .generator import CounterNarrativeGenerator, RebuttalResult
from .rag_retriever import RAGRetriever, RetrievedPassage

__all__ = [
    # Generator
    "CounterNarrativeGenerator",
    "RebuttalResult",
    # Retriever
    "RAGRetriever",
    "RetrievedPassage",
    # Formatter
    "PostFormatter",
    "FormattedPost",
]
