"""
ml/topic_model.py
=================
Zero-shot topic classifier for YouTube live-chat comments.
Enhanced with keyword fast-path for common Hinglish patterns.

Topics
------
  Appreciation  — praise, thanks, love, encouragement
  Question      — direct questions and doubts/confusion
  Promo         — self-promotion, links, "check my channel"
  Spam          — repeated noise, irrelevant flood, gibberish
  General       — anything that doesn't fit the above
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

from transformers import pipeline

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/bart-large-mnli"

VALID_TOPICS = {"Appreciation", "Question", "Promo", "Spam", "General"}

# More descriptive labels → better zero-shot accuracy
_CANDIDATE_LABELS = [
    "expressing appreciation, praise, love, or encouragement",
    "asking a question or expressing doubt, confusion, or not understanding something",
    "self-promotion, sharing links, or advertising a channel",
    "spam, gibberish, repeated characters, or completely irrelevant noise",
    "general neutral comment or statement",
]

_LABEL_MAP: dict[str, str] = {
    "expressing appreciation, praise, love, or encouragement":                        "Appreciation",
    "asking a question or expressing doubt, confusion, or not understanding something": "Question",
    "self-promotion, sharing links, or advertising a channel":                        "Promo",
    "spam, gibberish, repeated characters, or completely irrelevant noise":            "Spam",
    "general neutral comment or statement":                                            "General",
}

_MIN_CONFIDENCE = 0.40

# ── Keyword fast-path ──────────────────────────────────────────────────────────
_APPRECIATION_KW = {
    "love", "thanks", "thank", "superb", "amazing", "excellent", "great",
    "awesome", "wonderful", "brilliant", "fantastic", "best", "perfect",
    "mast", "zabardast", "kamaal", "jhakaas", "shandar", "lajawaab", "lajawab",
    "waah", "wah", "badhiya", "shukriya", "dhanyawad", "osm", "awsm",
    "dhansu", "pyaar", "acha", "accha", "bahut", "ekdum", "bindaas",
    "maja", "mazza", "dil", "khush", "happy", "nice", "good", "👏", "🙏", "😍", "💯",
}

_QUESTION_KW = {
    "kya", "kab", "kahan", "kaun", "kitna", "kitne", "konsa", "konsi",
    "kaise", "kyun", "kyunki",
    "what", "when", "where", "who", "which", "how", "why",
    "please explain", "bata", "batao", "bataye", "tell", "explain",
    # doubt/confusion words merged in
    "samajh", "confused", "confusion", "doubt", "unclear",
    "pata", "matlab", "matalab", "samjha", "samjhe", "samjhi", "smjh", "smjha",
    "nahi", "nhi", "nai",
}

_SPAM_PATTERNS = [
    r"(.)\1{4,}",           # yyyyyyy, aaaaaaa
    r"^[^a-zA-Z\u0900-\u097F]{0,3}$",  # only symbols/numbers/emojis, very short
]

_PROMO_KW = {
    "subscribe", "channel", "link", "follow", "instagram", "youtube",
    "check", "visit", "click", "http", "www", ".com", "telegram",
}


def _fast_path(text: str) -> tuple[str, float] | None:
    t = text.strip().lower()

    # Spam: repeated chars or gibberish
    for pat in _SPAM_PATTERNS:
        if re.search(pat, t):
            return "Spam", 0.85

    # Promo: links or channel promotion keywords
    if any(kw in t for kw in _PROMO_KW):
        return "Promo", 0.80

    words = set(t.split())

    # Question mark is a strong signal
    has_question_mark = "?" in text

    question_hits = len(words & _QUESTION_KW)
    appreciation_hits = len(words & _APPRECIATION_KW)

    # Appreciation wins if no question signals
    if appreciation_hits >= 1 and question_hits == 0 and not has_question_mark:
        return "Appreciation", min(0.72 + 0.05 * appreciation_hits, 0.92)

    if has_question_mark or question_hits >= 1:
        return "Question", min(0.75 + 0.04 * question_hits, 0.92)

    return None


# ── Lazy load ──────────────────────────────────────────────────────────────────
_lock       = threading.Lock()
_classifier = None


def _load_classifier():
    global _classifier
    if _classifier is not None:
        return
    with _lock:
        if _classifier is not None:
            return
        print("[topic] Loading zero-shot classifier…")
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=0 if _is_gpu_available() else -1,
        )
        print("[topic] Classifier ready ✓")


def _is_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Model prediction ───────────────────────────────────────────────────────────
@lru_cache(maxsize=512)
def _model_predict(text: str) -> tuple[str, float]:
    _load_classifier()
    result = _classifier(
        text,
        candidate_labels=_CANDIDATE_LABELS,
        multi_label=False,
    )
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    topic = _LABEL_MAP.get(top_label, "General")
    if top_score < _MIN_CONFIDENCE:
        topic = "General"

    return topic, round(top_score, 3)


# ── Public API ─────────────────────────────────────────────────────────────────
def predict_topic(text: str) -> tuple[str, float]:
    if not text or not text.strip():
        return "General", 0.5

    # Try fast-path first
    fast = _fast_path(text)
    if fast:
        return fast

    # Fall back to zero-shot model
    truncated = text.strip()[:256]
    topic, conf = _model_predict(truncated)

    if topic not in VALID_TOPICS:
        topic = "General"

    return topic, conf
