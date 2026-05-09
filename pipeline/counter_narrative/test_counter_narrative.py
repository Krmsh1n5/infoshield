"""
tests/test_counter_narrative.py

Test suite for pipeline/counter_narrative/.

Design principles
-----------------
- Zero real API calls: Anthropic client is fully mocked via unittest.mock.
- Deterministic: no network I/O, no filesystem writes outside tmp.
- Fast: all tests complete in < 2 s.
- Coverage: RAGRetriever, CounterNarrativeGenerator, PostFormatter.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sources_dir(tmp_path: Path) -> Path:
    """Create a minimal sources directory with two .txt files."""
    sources = tmp_path / "sources"
    sources.mkdir()

    (sources / "health_who.txt").write_text(
        "WHO confirms that 5G towers cannot spread viruses. "
        "Radio waves are non-ionising and cannot carry pathogens. "
        "Official information is available at https://www.euro.who.int/en/countries/azerbaijan.",
        encoding="utf-8",
    )
    (sources / "science_covid.txt").write_text(
        "ICNIRP guidelines confirm 5G exposure at regulated power levels poses no health risk. "
        "COVID-19 is caused by the SARS-CoV-2 virus, not by any radio technology. "
        "See https://www.icnirp.org for the full guidelines.",
        encoding="utf-8",
    )
    return sources


@pytest.fixture()
def mock_rebuttal_result():
    """A pre-built RebuttalResult for formatter tests."""
    from pipeline.counter_narrative.generator import RebuttalResult
    return RebuttalResult(
        text=(
            "Rəsmi WHO məlumatına görə, 5G şəbəkəsi virusları yaya bilməz. "
            "Elektromaqnit dalğaları bioloji agentləri daşımaq qabiliyyətinə malik deyil. "
            "Daha ətraflı məlumat üçün: https://www.euro.who.int/en/countries/azerbaijan"
        ),
        language="az",
        sources_used=["health_who", "science_covid"],
        confidence_in_rebuttal=0.92,
        topic="health",
        generation_time_ms=312,
    )


# ===========================================================================
# RAGRetriever tests
# ===========================================================================

class TestRAGRetriever:

    def test_loads_txt_files(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        retriever = RAGRetriever(sources_dir)
        assert retriever.num_chunks > 0

    def test_retrieve_returns_top_k(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        retriever = RAGRetriever(sources_dir)
        results = retriever.retrieve("5G and viruses", top_k=2)
        assert len(results) <= 2

    def test_retrieve_scores_are_descending(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        retriever = RAGRetriever(sources_dir)
        results = retriever.retrieve("COVID-19 radio waves health risk", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Scores must be descending"

    def test_retrieve_relevant_source_for_5g(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        retriever = RAGRetriever(sources_dir)
        results = retriever.retrieve("5G towers cause COVID-19", top_k=3)
        source_names = {r.source_file for r in results}
        # At least one of the relevant sources should surface
        assert source_names & {"health_who", "science_covid"}, (
            f"Expected relevant source, got: {source_names}"
        )

    def test_retrieve_as_context_returns_string_and_list(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        retriever = RAGRetriever(sources_dir)
        context, sources = retriever.retrieve_as_context("vaccine safety", top_k=2)
        assert isinstance(context, str)
        assert isinstance(sources, list)

    def test_empty_sources_dir_does_not_crash(self, tmp_path: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        empty_dir = tmp_path / "empty_sources"
        empty_dir.mkdir()
        retriever = RAGRetriever(empty_dir)
        assert retriever.num_chunks == 0
        results = retriever.retrieve("anything")
        assert results == []

    def test_comment_lines_excluded_from_chunks(self, tmp_path: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        d = tmp_path / "src"
        d.mkdir()
        (d / "test.txt").write_text(
            "# This is a comment line\n"
            "Vaccines are safe and effective. "
            "They undergo rigorous testing before approval.",
            encoding="utf-8",
        )
        retriever = RAGRetriever(d)
        # Chunks should not contain the comment text
        for chunk in retriever._chunks:
            assert "This is a comment line" not in chunk

    def test_passage_text_is_non_empty(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        retriever = RAGRetriever(sources_dir)
        for passage in retriever.retrieve("health", top_k=10):
            assert passage.text.strip(), "Passage text must not be empty"

    def test_min_score_filter(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.rag_retriever import RAGRetriever
        retriever = RAGRetriever(sources_dir)
        # A query with no match should yield nothing above min_score=0.99
        results = retriever.retrieve("banana smoothie recipe", top_k=5, min_score=0.99)
        assert all(r.score >= 0.99 for r in results)


# ===========================================================================
# CounterNarrativeGenerator tests  (Anthropic API fully mocked)
# ===========================================================================

def _make_mock_anthropic_response(text: str):
    """Build a minimal mock that mimics anthropic.Anthropic().messages.create()."""
    content_block = MagicMock()
    content_block.text = text

    response = MagicMock()
    response.content = [content_block]
    return response


class TestCounterNarrativeGenerator:

    @pytest.fixture()
    def generator(self, sources_dir: Path):
        """Instantiate generator with mocked Anthropic client."""
        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            from pipeline.counter_narrative.generator import CounterNarrativeGenerator
            gen = CounterNarrativeGenerator(
                model="claude-sonnet-4-6",
                sources_dir=sources_dir,
                api_key="test-key-not-real",
            )
            gen._client = mock_client   # attach so tests can configure .create()
            yield gen

    def test_generate_rebuttal_returns_result(self, generator) -> None:
        from pipeline.counter_narrative.generator import RebuttalResult
        generator._client.messages.create.return_value = _make_mock_anthropic_response(
            "5G texnologiyası virus yaymır. "
            "WHO bu iddianın yanlış olduğunu təsdiqləyir. "
            "https://www.euro.who.int/en/countries/azerbaijan\n"
            "CONFIDENCE: 0.91"
        )
        result = generator.generate_rebuttal(
            false_claim="5G spreads COVID-19",
            topic="health",
            language="az",
        )
        assert isinstance(result, RebuttalResult)

    def test_rebuttal_language_preserved(self, generator) -> None:
        for lang in ("az", "ru", "en"):
            generator._client.messages.create.return_value = _make_mock_anthropic_response(
                f"Rebuttal in {lang}.\nCONFIDENCE: 0.85"
            )
            result = generator.generate_rebuttal(
                false_claim="vaccines contain microchips",
                topic="vaccine",
                language=lang,  # type: ignore[arg-type]
            )
            assert result.language == lang

    def test_confidence_parsed_correctly(self, generator) -> None:
        generator._client.messages.create.return_value = _make_mock_anthropic_response(
            "Vaccines are safe per WHO data. "
            "No microchips have been found in any approved vaccine. "
            "https://www.sehiyyenazirligi.gov.az\n"
            "CONFIDENCE: 0.88"
        )
        result = generator.generate_rebuttal(
            false_claim="vaccines have microchips",
            topic="vaccine",
            language="en",
        )
        assert abs(result.confidence_in_rebuttal - 0.88) < 1e-6

    def test_confidence_clamped_to_0_1(self, generator) -> None:
        generator._client.messages.create.return_value = _make_mock_anthropic_response(
            "Some rebuttal.\nCONFIDENCE: 1.5"
        )
        result = generator.generate_rebuttal("claim", language="en")
        assert result.confidence_in_rebuttal <= 1.0

    def test_sources_used_populated(self, generator) -> None:
        generator._client.messages.create.return_value = _make_mock_anthropic_response(
            "Valid rebuttal text here.\nCONFIDENCE: 0.75"
        )
        result = generator.generate_rebuttal(
            false_claim="5G towers spread COVID",
            topic="health",
            language="en",
        )
        assert isinstance(result.sources_used, list)

    def test_api_error_yields_fallback(self, generator) -> None:
        import anthropic as _anthropic
        generator._client.messages.create.side_effect = _anthropic.APIError(
            message="rate limited", request=MagicMock(), body=None
        )
        result = generator.generate_rebuttal(
            false_claim="election was rigged",
            topic="election",
            language="en",
        )
        assert result.is_fallback() or len(result.text) > 0  # graceful degradation
        assert result.sources_used == []

    def test_generation_time_recorded(self, generator) -> None:
        generator._client.messages.create.return_value = _make_mock_anthropic_response(
            "Rebuttal.\nCONFIDENCE: 0.7"
        )
        result = generator.generate_rebuttal("false claim", language="en")
        assert result.generation_time_ms >= 0

    def test_topic_preserved_in_result(self, generator) -> None:
        generator._client.messages.create.return_value = _make_mock_anthropic_response(
            "Military rebuttal.\nCONFIDENCE: 0.80"
        )
        result = generator.generate_rebuttal(
            false_claim="army retreated",
            topic="military",
            language="ru",
        )
        assert result.topic == "military"

    def test_missing_api_key_raises(self, sources_dir: Path) -> None:
        import os
        with patch.dict(os.environ, {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is absent
            os.environ.pop("ANTHROPIC_API_KEY", None)
            from pipeline.counter_narrative.generator import CounterNarrativeGenerator
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                CounterNarrativeGenerator(sources_dir=sources_dir)

    def test_rebuttal_text_not_empty(self, generator) -> None:
        generator._client.messages.create.return_value = _make_mock_anthropic_response(
            "This claim has no basis in evidence per WHO.\nCONFIDENCE: 0.9"
        )
        result = generator.generate_rebuttal("fabricated health claim", language="en")
        assert result.text.strip()


# ===========================================================================
# PostFormatter tests
# ===========================================================================

class TestPostFormatter:

    @pytest.fixture()
    def formatter(self):
        from pipeline.counter_narrative.formatter import PostFormatter
        return PostFormatter()

    # --- Telegram ---

    def test_telegram_within_limit(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "telegram")
        assert len(output) <= 1024, f"Telegram output too long: {len(output)}"

    def test_telegram_has_pin_emoji(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "telegram")
        assert "📌" in output

    def test_telegram_has_html_bold(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "telegram")
        assert "<b>" in output and "</b>" in output

    def test_telegram_long_text_truncated(self, formatter) -> None:
        from pipeline.counter_narrative.generator import RebuttalResult
        long_result = RebuttalResult(
            text="X" * 2000,
            language="en",
            sources_used=[],
            confidence_in_rebuttal=0.8,
            topic="health",
            generation_time_ms=100,
        )
        output = formatter.format_for_platform(long_result, "telegram")
        assert len(output) <= 1024

    # --- Instagram ---

    def test_instagram_within_limit(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "instagram")
        assert len(output) <= 300, f"Instagram output too long: {len(output)}"

    def test_instagram_has_hashtags_when_space_allows(self, formatter) -> None:
        from pipeline.counter_narrative.generator import RebuttalResult
        short_result = RebuttalResult(
            text="Short factual rebuttal.",
            language="en",
            sources_used=["health_who"],
            confidence_in_rebuttal=0.9,
            topic="health",
            generation_time_ms=50,
        )
        output = formatter.format_for_platform(short_result, "instagram")
        assert "#" in output

    def test_instagram_no_html_tags(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "instagram")
        assert "<b>" not in output and "<a " not in output

    def test_instagram_long_text_truncated(self, formatter) -> None:
        from pipeline.counter_narrative.generator import RebuttalResult
        long_result = RebuttalResult(
            text="Y" * 500,
            language="en",
            sources_used=[],
            confidence_in_rebuttal=0.7,
            topic="science",
            generation_time_ms=200,
        )
        output = formatter.format_for_platform(long_result, "instagram")
        assert len(output) <= 300

    # --- WhatsApp ---

    def test_whatsapp_within_limit(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "whatsapp")
        assert len(output) <= 500, f"WhatsApp output too long: {len(output)}"

    def test_whatsapp_plain_text(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "whatsapp")
        assert "<b>" not in output
        assert "#" not in output

    def test_whatsapp_long_text_truncated(self, formatter) -> None:
        from pipeline.counter_narrative.generator import RebuttalResult
        long_result = RebuttalResult(
            text="Z" * 1000,
            language="ru",
            sources_used=[],
            confidence_in_rebuttal=0.6,
            topic="military",
            generation_time_ms=150,
        )
        output = formatter.format_for_platform(long_result, "whatsapp")
        assert len(output) <= 500

    # --- Facebook ---

    def test_facebook_within_limit(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "facebook")
        assert len(output) <= 2000, f"Facebook output too long: {len(output)}"

    def test_facebook_has_footer(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "facebook")
        assert "📢" in output

    def test_facebook_no_html(self, formatter, mock_rebuttal_result) -> None:
        output = formatter.format_for_platform(mock_rebuttal_result, "facebook")
        assert "<b>" not in output

    # --- Cross-platform ---

    def test_all_platforms_return_strings(self, formatter, mock_rebuttal_result) -> None:
        for platform in ("telegram", "instagram", "whatsapp", "facebook"):
            out = formatter.format_for_platform(
                mock_rebuttal_result, platform  # type: ignore[arg-type]
            )
            assert isinstance(out, str), f"Expected str for {platform}"
            assert out.strip(), f"Output must not be empty for {platform}"

    def test_unknown_platform_raises_value_error(self, formatter, mock_rebuttal_result) -> None:
        with pytest.raises(ValueError, match="Unknown platform"):
            formatter.format_for_platform(
                mock_rebuttal_result,
                "tiktok",  # type: ignore[arg-type]
            )

    def test_format_all_platforms_returns_all_keys(self, formatter, mock_rebuttal_result) -> None:
        results = formatter.format_all_platforms(mock_rebuttal_result)
        assert set(results.keys()) == {"telegram", "instagram", "whatsapp", "facebook"}

    def test_multilingual_formatting_az(self, formatter) -> None:
        from pipeline.counter_narrative.generator import RebuttalResult
        az_result = RebuttalResult(
            text="Bu iddia yanlışdır.",
            language="az",
            sources_used=["health_who"],
            confidence_in_rebuttal=0.88,
            topic="health",
            generation_time_ms=200,
        )
        tg = formatter.format_for_platform(az_result, "telegram")
        assert "RƏSMİ CAVAB" in tg

    def test_multilingual_formatting_ru(self, formatter) -> None:
        from pipeline.counter_narrative.generator import RebuttalResult
        ru_result = RebuttalResult(
            text="Это утверждение ложно.",
            language="ru",
            sources_used=["health_moh"],
            confidence_in_rebuttal=0.85,
            topic="health",
            generation_time_ms=180,
        )
        tg = formatter.format_for_platform(ru_result, "telegram")
        assert "ОФИЦИАЛЬНЫЙ ОТВЕТ" in tg

    def test_sources_cited_in_rebuttal_text(self, formatter) -> None:
        """Ensure the generated text contains a URL (source citation)."""
        from pipeline.counter_narrative.generator import RebuttalResult
        result = RebuttalResult(
            text=(
                "WHO data shows no link between 5G and COVID-19. "
                "See https://www.euro.who.int/en/countries/azerbaijan"
            ),
            language="en",
            sources_used=["health_who"],
            confidence_in_rebuttal=0.93,
            topic="health",
            generation_time_ms=100,
        )
        for platform in ("telegram", "whatsapp", "facebook"):
            out = formatter.format_for_platform(result, platform)  # type: ignore[arg-type]
            assert "http" in out, f"URL missing in {platform} output"


# ===========================================================================
# Integration smoke test (no API call)
# ===========================================================================

class TestIntegrationSmoke:
    """Verify the full pipeline: retrieve → generate (mocked) → format."""

    def test_full_pipeline_smoke(self, sources_dir: Path) -> None:
        from pipeline.counter_narrative.formatter import PostFormatter
        from pipeline.counter_narrative.generator import CounterNarrativeGenerator

        with patch("anthropic.Anthropic") as MockAnthropic:
            mock_client = MagicMock()
            MockAnthropic.return_value = mock_client
            mock_client.messages.create.return_value = _make_mock_anthropic_response(
                "5G şəbəkəsi virus yaymır. "
                "Bu elmi cəhətdən mümkün deyil. "
                "https://www.euro.who.int/en/countries/azerbaijan\n"
                "CONFIDENCE: 0.94"
            )

            gen = CounterNarrativeGenerator(
                sources_dir=sources_dir,
                api_key="test-key",
            )
            result = gen.generate_rebuttal(
                false_claim="5G towers spread COVID-19",
                topic="health",
                language="az",
                confidence=0.92,
                cascade_pattern="wide_burst",
            )

        formatter = PostFormatter()
        for platform in ("telegram", "instagram", "whatsapp", "facebook"):
            post = formatter.format_for_platform(result, platform)  # type: ignore[arg-type]
            limit = {"telegram": 1024, "instagram": 300, "whatsapp": 500, "facebook": 2000}[platform]
            assert len(post) <= limit, f"{platform} exceeds limit: {len(post)}"
            assert post.strip()
