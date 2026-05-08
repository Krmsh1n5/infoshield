"""
pipeline/counter_narrative/formatter.py

Platform-aware post formatter for InfoShield counter-narratives.

Each platform has distinct:
  - Character limit
  - Formatting support (HTML / plain text)
  - Cultural conventions (hashtags, emoji placement)

Supported platforms: "telegram" | "instagram" | "whatsapp" | "facebook"
"""

from __future__ import annotations

import html
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Literal

from .generator import RebuttalResult

logger = logging.getLogger(__name__)

Platform = Literal["telegram", "instagram", "whatsapp", "facebook"]

# ---------------------------------------------------------------------------
# Platform limits (hard chars, Unicode-safe)
# ---------------------------------------------------------------------------

_LIMITS: dict[str, int] = {
    "telegram":  1024,
    "instagram":  300,
    "whatsapp":   500,
    "facebook":  2000,
}

# ---------------------------------------------------------------------------
# Hashtag bank (language → topic → tags)
# ---------------------------------------------------------------------------

_HASHTAGS: dict[str, dict[str, list[str]]] = {
    "az": {
        "health":    ["#Sağlamlıq", "#RəsmiMəlumat", "#ÜST"],
        "vaccine":   ["#Peyvənd", "#RəsmiMəlumat", "#SəhiyyəNazirlyi"],
        "covid":     ["#COVID19", "#RəsmiMəlumat", "#ÜST"],
        "5g":        ["#5G", "#Faktlar", "#ElmiMəlumat"],
        "military":  ["#RəsmiMəlumat", "#MüdafiəNazirlyi"],
        "election":  ["#MSK", "#RəsmiMəlumat", "#Seçkilər"],
        "science":   ["#Elm", "#Faktlar", "#RəsmiMəlumat"],
        "default":   ["#RəsmiMəlumat", "#Faktlar"],
    },
    "ru": {
        "health":    ["#Здоровье", "#ОфициальноеСообщение", "#ВОЗ"],
        "vaccine":   ["#Вакцинация", "#ОфициальноеСообщение"],
        "covid":     ["#COVID19", "#ОфициальноеСообщение", "#ВОЗ"],
        "5g":        ["#5G", "#Факты", "#НаучныеДанные"],
        "military":  ["#ОфициальноеСообщение", "#МинистерствоОбороны"],
        "election":  ["#ЦИК", "#ОфициальноеСообщение", "#Выборы"],
        "science":   ["#Наука", "#Факты", "#ОфициальноеСообщение"],
        "default":   ["#ОфициальноеСообщение", "#Факты"],
    },
    "en": {
        "health":    ["#Health", "#OfficialInfo", "#WHO"],
        "vaccine":   ["#Vaccine", "#OfficialInfo", "#VaccineFactCheck"],
        "covid":     ["#COVID19", "#OfficialInfo", "#WHO"],
        "5g":        ["#5G", "#FactCheck", "#Science"],
        "military":  ["#OfficialInfo", "#MoD"],
        "election":  ["#ElectionIntegrity", "#OfficialInfo", "#FactCheck"],
        "science":   ["#Science", "#FactCheck", "#OfficialInfo"],
        "default":   ["#OfficialInfo", "#FactCheck"],
    },
}


def _get_hashtags(language: str, topic: str) -> list[str]:
    lang_tags = _HASHTAGS.get(language, _HASHTAGS["en"])
    return lang_tags.get(topic, lang_tags["default"])


# ---------------------------------------------------------------------------
# Truncation utilities
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int, ellipsis: str = "…") -> str:
    """Hard-truncate to max_chars, appending ellipsis only if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ellipsis)] + ellipsis


def _safe_truncate_words(text: str, max_chars: int, ellipsis: str = "…") -> str:
    """Word-boundary truncation — avoids splitting in mid-word."""
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - len(ellipsis)]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + ellipsis


# ---------------------------------------------------------------------------
# Per-platform formatters
# ---------------------------------------------------------------------------

def _format_telegram(rebuttal: RebuttalResult) -> str:
    """
    Telegram: HTML formatting, max 1 024 chars, 📌 header, bold label.

    Telegram supports a limited HTML subset:
    <b>, <i>, <u>, <s>, <a href="...">, <code>, <pre>
    """
    # Reserve space: header ~40, footer label ~30
    body_limit = _LIMITS["telegram"] - 80

    escaped_body = html.escape(rebuttal.text)
    truncated_body = _safe_truncate_words(escaped_body, body_limit)

    label_map = {"az": "RƏSMİ CAVAB", "ru": "ОФИЦИАЛЬНЫЙ ОТВЕТ", "en": "OFFICIAL REBUTTAL"}
    label = label_map.get(rebuttal.language, label_map["en"])

    output = (
        f"📌 <b>{label}</b>\n\n"
        f"{truncated_body}"
    )
    # Final hard cap
    return _truncate(output, _LIMITS["telegram"])


def _format_instagram(rebuttal: RebuttalResult) -> str:
    """
    Instagram: plain text, max 300 chars, hashtags appended.

    Instagram does not render HTML. Hashtags are added only if space permits.
    """
    tags = _get_hashtags(rebuttal.language, rebuttal.topic)
    tags_str = " ".join(tags)
    # Reserve space for tags + newline separator
    tags_portion = f"\n\n{tags_str}"
    body_limit = _LIMITS["instagram"] - len(tags_portion)

    body = _safe_truncate_words(rebuttal.text, max(body_limit, 60))
    candidate = body + tags_portion

    if len(candidate) <= _LIMITS["instagram"]:
        return candidate

    # Tags don't fit — drop hashtags, keep clean body
    return _truncate(rebuttal.text, _LIMITS["instagram"])


def _format_whatsapp(rebuttal: RebuttalResult) -> str:
    """
    WhatsApp: plain text, max 500 chars, no special formatting.

    WhatsApp renders *bold* and _italic_ but we keep it clean/neutral
    for government public communications tone.
    """
    return _safe_truncate_words(rebuttal.text, _LIMITS["whatsapp"])


def _format_facebook(rebuttal: RebuttalResult) -> str:
    """
    Facebook: plain text with line breaks, max 2 000 chars.

    Adds a one-line contextual footer with topic label.
    """
    label_map = {
        "az": "📢 Rəsmi məlumat",
        "ru": "📢 Официальная информация",
        "en": "📢 Official information",
    }
    label = label_map.get(rebuttal.language, label_map["en"])

    # Wrap long paragraphs at ~80 chars for readability
    wrapped = textwrap.fill(rebuttal.text, width=80)

    footer = f"\n\n{label}"
    body_limit = _LIMITS["facebook"] - len(footer)
    truncated = _safe_truncate_words(wrapped, body_limit)

    return truncated + footer


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FORMATTERS = {
    "telegram":  _format_telegram,
    "instagram": _format_instagram,
    "whatsapp":  _format_whatsapp,
    "facebook":  _format_facebook,
}


@dataclass
class FormattedPost:
    """Result of PostFormatter.format_for_platform."""
    platform: Platform
    text: str
    char_count: int = field(init=False)
    within_limit: bool = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        self.within_limit = self.char_count <= _LIMITS[self.platform]


class PostFormatter:
    """
    Formats a RebuttalResult for a target social/messaging platform.

    Usage
    -----
    formatter = PostFormatter()
    post = formatter.format_for_platform(result, "telegram")
    print(post.text)
    assert post.within_limit
    """

    def format_for_platform(
        self,
        rebuttal: RebuttalResult,
        platform: Platform,
    ) -> str:
        """
        Format *rebuttal* for the given platform.

        Parameters
        ----------
        rebuttal : RebuttalResult from CounterNarrativeGenerator
        platform : "telegram" | "instagram" | "whatsapp" | "facebook"

        Returns
        -------
        Formatted string guaranteed to be within the platform's character limit.

        Raises
        ------
        ValueError if *platform* is not recognised.
        """
        formatter = _FORMATTERS.get(platform)
        if formatter is None:
            raise ValueError(
                f"Unknown platform '{platform}'. "
                f"Valid options: {sorted(_FORMATTERS.keys())}"
            )

        formatted = formatter(rebuttal)

        limit = _LIMITS[platform]
        if len(formatted) > limit:
            # Final safety net — should not be reached given per-formatter logic
            logger.warning(
                "PostFormatter: %s output exceeded limit (%d > %d), hard-truncating.",
                platform, len(formatted), limit,
            )
            formatted = _truncate(formatted, limit)

        logger.debug(
            "PostFormatter: platform=%s lang=%s chars=%d/%d",
            platform, rebuttal.language, len(formatted), limit,
        )
        return formatted

    def format_all_platforms(
        self,
        rebuttal: RebuttalResult,
    ) -> dict[Platform, str]:
        """
        Convenience method: format for all supported platforms at once.

        Returns
        -------
        Dict mapping platform name → formatted string.
        """
        return {
            platform: self.format_for_platform(rebuttal, platform)  # type: ignore[arg-type]
            for platform in _FORMATTERS
        }
