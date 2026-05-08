"""
api/routes/rebuttal.py
======================
POST /api/v1/rebuttal

Generates a RAG-grounded counter-narrative and formats it for
Telegram, Instagram, and WhatsApp.
"""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

log = logging.getLogger(__name__)

rebuttal_bp = Blueprint("rebuttal", __name__)

_FALLBACK_TEXTS = {
    "az": "Bu iddia araşdırılır. Rəsmi mənbəyə müraciət edin.",
    "ru": "Данное утверждение проверяется. Обратитесь к официальному источнику.",
    "en": "This claim is under investigation. Please refer to official sources.",
}


def _mock_rebuttal(claim: str, topic: str, language: str) -> dict:
    text = _FALLBACK_TEXTS.get(language, _FALLBACK_TEXTS["en"])
    return {
        "rebuttal": text,
        "language": language,
        "topic": topic,
        "sources": [],
        "confidence_in_rebuttal": 0.0,
        "generation_time_ms": 0,
        "formatted": {
            "telegram": f"📌 <b>OFFICIAL REBUTTAL</b>\n\n{text}",
            "instagram": text,
            "whatsapp": text,
        },
        "mock": True,
    }


@rebuttal_bp.route("/api/v1/rebuttal", methods=["POST"])
def rebuttal():
    body = request.get_json(force=True, silent=True) or {}
    claim = body.get("claim", "").strip()
    topic = body.get("topic", "health")
    language = body.get("language", "az")
    confidence = float(body.get("confidence", 0.9))

    if not claim:
        return jsonify({"error": "Bad request", "code": 400,
                        "detail": "'claim' field is required."}), 400

    if language not in ("az", "ru", "en"):
        return jsonify({"error": "Bad request", "code": 400,
                        "detail": "language must be 'az', 'ru', or 'en'."}), 400

    generator = current_app.extensions.get("counter_narrative_generator")
    formatter = current_app.extensions.get("post_formatter")

    if generator is None or formatter is None:
        log.warning("CounterNarrativeGenerator not available — returning mock rebuttal")
        return jsonify(_mock_rebuttal(claim, topic, language))

    try:
        result = generator.generate_rebuttal(
            false_claim=claim,
            topic=topic,
            language=language,
            confidence=confidence,
            cascade_pattern="wide_burst",
        )

        formatted = formatter.format_all_platforms(result)

        return jsonify({
            "rebuttal": result.text,
            "language": result.language,
            "topic": result.topic,
            "sources": result.sources_used,
            "confidence_in_rebuttal": result.confidence_in_rebuttal,
            "generation_time_ms": result.generation_time_ms,
            "formatted": {
                "telegram": formatted.get("telegram", result.text),
                "instagram": formatted.get("instagram", result.text),
                "whatsapp": formatted.get("whatsapp", result.text),
            },
            "mock": False,
        })

    except Exception as exc:
        log.exception("rebuttal generation error")
        return jsonify({"error": "Rebuttal generation failed", "code": 500,
                        "detail": str(exc)}), 500
