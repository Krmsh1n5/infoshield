"""
pipeline/counter_narrative/generator.py

RAG-grounded counter-narrative generator for InfoShield.

Produces short, factual rebuttals in Azerbaijani (az), Russian (ru),
or English (en) using the Anthropic Messages API with verified source
passages as grounding context.

Architecture
------------
  false_claim
      │
      ▼
  RAGRetriever  ──── top-k passages
      │
      ▼
  CounterNarrativeGenerator
      │  builds prompt
      ▼
  claude-sonnet-4-6
      │  3-sentence rebuttal
      ▼
  RebuttalResult
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic

from .rag_retriever import RAGRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Language = Literal["az", "ru", "en"]
CascadePattern = Literal["wide_burst", "deep_chain", "slow_diffusion", "unknown"]

_LANGUAGE_NAMES: dict[str, str] = {
    "az": "Azerbaijani",
    "ru": "Russian",
    "en": "English",
}

_OFFICIAL_SOURCES: dict[str, str] = {
    "health":   "https://www.sehiyyenazirligi.gov.az",
    "who":      "https://www.euro.who.int/en/countries/azerbaijan",
    "military": "https://mod.gov.az",
    "election": "https://www.msk.gov.az",
    "science":  "https://www.who.int/news-room/feature-stories",
    "default":  "https://president.az/en",
}

_TOPIC_SOURCE_MAP: dict[str, str] = {
    "health":   "health",
    "vaccine":  "health",
    "covid":    "science",
    "5g":       "science",
    "military": "military",
    "defence":  "military",
    "election": "election",
    "vote":     "election",
    "science":  "science",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RebuttalResult:
    """Structured output from CounterNarrativeGenerator.generate_rebuttal."""
    text: str
    language: Language
    sources_used: list[str] = field(default_factory=list)
    confidence_in_rebuttal: float = 0.0
    topic: str = "general"
    generation_time_ms: int = 0

    def is_fallback(self) -> bool:
        """True if the generator fell back to an under-investigation response."""
        return "under investigation" in self.text.lower() or \
               "araşdırılır" in self.text.lower() or \
               "расследуется" in self.text.lower()


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a fact-checking assistant for a public information service. \
Produce a SHORT rebuttal in {language_name} for the detected false claim.

Rules:
- Maximum 3 sentences
- Ground every fact in the provided verified sources
- Never repeat the false claim directly
- Tone: calm, informative, not condescending
- End with one source URL
- If sources do not contain enough information, say: \
'This claim is under investigation. Check [official source] for updates.'
Do NOT invent facts.

After the rebuttal, on a NEW LINE, write exactly:
CONFIDENCE: <float between 0.0 and 1.0>
"""

_USER_PROMPT_TEMPLATE = """\
DETECTED FALSE CLAIM: {false_claim}

TOPIC CATEGORY: {topic}
SPREAD PATTERN: {cascade_pattern}
DETECTION CONFIDENCE: {confidence:.2f}

{context}

Produce the rebuttal now."""


def _build_system_prompt(language: Language) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        language_name=_LANGUAGE_NAMES.get(language, "English")
    )


def _build_user_prompt(
    false_claim: str,
    topic: str,
    confidence: float,
    cascade_pattern: str,
    context: str,
) -> str:
    return _USER_PROMPT_TEMPLATE.format(
        false_claim=false_claim,
        topic=topic,
        cascade_pattern=cascade_pattern,
        confidence=confidence,
        context=context,
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class CounterNarrativeGenerator:
    """
    Generates RAG-grounded counter-narratives via the Anthropic Messages API.

    Parameters
    ----------
    model       : Anthropic model ID (default: claude-sonnet-4-6)
    sources_dir : Path to directory containing verified .txt source files.
                  Defaults to  <this file's directory>/sources/
    api_key     : Anthropic API key.  Falls back to ANTHROPIC_API_KEY env var.

    Usage
    -----
    gen = CounterNarrativeGenerator()
    result = gen.generate_rebuttal(
        false_claim="5G towers spread COVID-19",
        topic="health",
        language="az",
        confidence=0.92,
        cascade_pattern="wide_burst",
    )
    print(result.text)
    """

    _DEFAULT_SOURCES_DIR = Path(__file__).parent / "sources"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        sources_dir: Path | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.sources_dir = Path(sources_dir or self._DEFAULT_SOURCES_DIR)

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it or pass api_key= to CounterNarrativeGenerator()."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._retriever = RAGRetriever(self.sources_dir)

        logger.info(
            "CounterNarrativeGenerator ready: model=%s, sources=%d chunks",
            self.model,
            self._retriever.num_chunks,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_llm_response(raw: str) -> tuple[str, float]:
        """
        Parse raw LLM output into (rebuttal_text, confidence).

        The model is instructed to append:
            CONFIDENCE: 0.85
        on the last line.
        """
        lines = raw.strip().splitlines()
        confidence = 0.5
        rebuttal_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(stripped.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            else:
                rebuttal_lines.append(line)

        rebuttal = "\n".join(rebuttal_lines).strip()
        return rebuttal, confidence

    @staticmethod
    def _fallback_response(language: Language, topic: str) -> str:
        """Return a safe under-investigation message in the requested language."""
        official_url = _OFFICIAL_SOURCES.get(
            _TOPIC_SOURCE_MAP.get(topic, "default"), _OFFICIAL_SOURCES["default"]
        )
        messages = {
            "az": (
                f"Bu iddia araşdırılır. "
                f"Yenilənmiş məlumat üçün rəsmi mənbəyə baxın: {official_url}"
            ),
            "ru": (
                f"Данное утверждение находится на проверке. "
                f"Актуальная информация: {official_url}"
            ),
            "en": (
                f"This claim is under investigation. "
                f"Check the official source for updates: {official_url}"
            ),
        }
        return messages.get(language, messages["en"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_rebuttal(
        self,
        false_claim: str,
        topic: str = "health",
        language: Language = "az",
        confidence: float = 0.9,
        cascade_pattern: CascadePattern = "wide_burst",
        top_k_sources: int = 3,
    ) -> RebuttalResult:
        """
        Generate a grounded counter-narrative rebuttal.

        Parameters
        ----------
        false_claim     : The misinformation text detected by BiGCN/SBM pipeline.
        topic           : Semantic category ("health", "military", "election", …).
        language        : Output language: "az" | "ru" | "en".
        confidence      : BiGCN detection confidence (passed to prompt for context).
        cascade_pattern : Spread pattern label from pipeline analysis.
        top_k_sources   : Number of RAG passages to retrieve.

        Returns
        -------
        RebuttalResult with text, language, sources_used, confidence, timing.
        """
        start_ms = int(time.time() * 1000)

        # 1. Retrieve grounding context
        context, sources_used = self._retriever.retrieve_as_context(
            query=false_claim,
            top_k=top_k_sources,
        )

        # 2. Build prompts
        system_prompt = _build_system_prompt(language)
        user_prompt = _build_user_prompt(
            false_claim=false_claim,
            topic=topic,
            confidence=confidence,
            cascade_pattern=cascade_pattern,
            context=context,
        )

        # 3. Call Anthropic API
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw_text = response.content[0].text
            rebuttal_text, llm_confidence = self._parse_llm_response(raw_text)

        except anthropic.APIError as exc:
            logger.error("Anthropic API error: %s", exc)
            rebuttal_text = self._fallback_response(language, topic)
            llm_confidence = 0.0
            sources_used = []

        except Exception as exc:
            logger.error("Unexpected error in generate_rebuttal: %s", exc)
            rebuttal_text = self._fallback_response(language, topic)
            llm_confidence = 0.0
            sources_used = []

        end_ms = int(time.time() * 1000)

        return RebuttalResult(
            text=rebuttal_text,
            language=language,
            sources_used=sources_used,
            confidence_in_rebuttal=llm_confidence,
            topic=topic,
            generation_time_ms=end_ms - start_ms,
        )
