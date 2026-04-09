from __future__ import annotations

import re
import threading
from functools import lru_cache

import emoji
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── Model paths ────────────────────────────────────────────────────────────────
MURIL_MODEL    = "./new_trained_data/muril-sentimix"
XLMR_MODEL     = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MULTI_MODEL    = "tabularisai/multilingual-sentiment-analysis"

LABELS = ["Negative", "Neutral", "Positive"]

# Weights
MURIL_WEIGHT = 0.40
XLMR_WEIGHT  = 0.35
MULTI_WEIGHT = 0.25


# ── Lazy loading ───────────────────────────────────────────────────────────────
_lock = threading.Lock()

_muril_tokenizer = _muril_model = None
_xlmr_tokenizer  = _xlmr_model  = None
_multi_tokenizer = _multi_model = None
_models_loaded   = False
_load_error: Exception | None = None


def _load_models():
    global _muril_tokenizer, _muril_model
    global _xlmr_tokenizer, _xlmr_model
    global _multi_tokenizer, _multi_model
    global _models_loaded, _load_error

    if _models_loaded:
        return

    with _lock:
        if _models_loaded:
            return

        print("[sentiment] Loading models...")
        try:
            _muril_tokenizer = AutoTokenizer.from_pretrained(MURIL_MODEL)
            _muril_model     = AutoModelForSequenceClassification.from_pretrained(MURIL_MODEL)
            print(f"[sentiment] MuRIL loaded — id2label: {_muril_model.config.id2label}")

            _xlmr_tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL)
            _xlmr_model     = AutoModelForSequenceClassification.from_pretrained(XLMR_MODEL)
            print(f"[sentiment] XLM-R loaded — id2label: {_xlmr_model.config.id2label}")

            _multi_tokenizer = AutoTokenizer.from_pretrained(MULTI_MODEL)
            _multi_model     = AutoModelForSequenceClassification.from_pretrained(MULTI_MODEL)
            print(f"[sentiment] Multilingual loaded — id2label: {_multi_model.config.id2label}")

            _muril_model.eval()
            _xlmr_model.eval()
            _multi_model.eval()

            if torch.cuda.is_available():
                _muril_model.to("cuda")
                _xlmr_model.to("cuda")
                _multi_model.to("cuda")

            _models_loaded = True
            print("[sentiment] All models ready ✓")

        except Exception as exc:
            _load_error = exc
            print(f"[sentiment] ERROR loading models: {exc}")
            raise


def _device():
    if not _models_loaded:
        _load_models()
    return next(_muril_model.parameters()).device


# ── Text normalization ─────────────────────────────────────────────────────────
def _normalize_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


# ── Emoji scoring ──────────────────────────────────────────────────────────────
_POS_KW = {"love", "fire", "happy", "laugh", "win", "cool", "best", "heart", "smile", "star", "clap", "pray", "sparkle", "sun", "rainbow"}
_NEG_KW = {"angry", "sad", "cry", "worst", "bad", "hate", "skull", "vomit", "rage", "broken", "disappointed"}


def _emoji_score(text: str):
    score = 0
    for ch in text:
        if emoji.is_emoji(ch):
            name = emoji.demojize(ch)
            if any(k in name for k in _POS_KW):
                score += 0.2
            elif any(k in name for k in _NEG_KW):
                score -= 0.2
    return score


# ── Hinglish slang ─────────────────────────────────────────────────────────────
_SLANG = {
    # Positive
    "mast":       "excellent",
    "op":         "excellent",
    "lit":        "amazing",
    "sahi":       "correct good",
    "jhakaas":    "awesome",
    "kadak":      "strong good",
    "zabardast":  "fantastic",
    "kamaal":     "amazing",
    "bindaas":    "great",
    "ekdum":      "absolutely",
    "shandar":    "splendid",
    "lajawaab":   "outstanding",
    "waah":       "wow great",
    "wah":        "wow great",
    "superb":     "excellent",
    "osm":        "awesome",
    "awsm":       "awesome",
    "gr8":        "great",
    "lajawab":    "outstanding",
    "dhansu":     "awesome",
    "fatafat":    "excellent quick",
    "mazza":      "fun enjoyable",
    "maja":       "fun enjoyable",
    "acha":       "good",
    "accha":      "good",
    "badhiya":    "very good",
    "shukriya":   "thank you grateful",
    "dhanyawad":  "thank you grateful",
    "love":       "love positive",
    "pyaar":      "love positive",

    # Negative
    "bakwas":     "nonsense bad",
    "faltu":      "useless bad",
    "bekar":      "useless bad",
    "ghatiya":    "terrible bad",
    "wahiyat":    "awful bad",
    "bura":       "bad negative",
    "kharab":     "bad negative",
    "boring":     "boring negative",
    "bekaar":     "useless bad",
    "chutiya":    "stupid offensive",
    "ullu":       "fool negative",
    "pagal":      "crazy negative",
    "besharam":   "shameless negative",
    "nafrat":     "hate negative",
    "gussa":      "angry negative",
    "naraaz":     "angry upset",
    "dukh":       "sad negative",
    "takleef":    "pain negative",
    "mushkil":    "difficult negative",
    "problem":    "problem negative",
}


def _preprocess(text: str) -> str:
    text = _normalize_repeated_chars(text)

    text = emoji.replace_emoji(
        text,
        replace=lambda ch, data_dict: f" {emoji.demojize(ch).strip(':')} " if emoji.is_emoji(ch) else ch
    )

    text = text.lower()

    words = []
    for w in text.split():
        if w in _SLANG:
            words.append(_SLANG[w])
        else:
            words.append(w)

    text = " ".join(words)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


# ── Fast path ──────────────────────────────────────────────────────────────────
_POS_SLANG = {"mast", "op", "lit", "sahi", "jhakaas", "kadak", "zabardast", "kamaal",
              "bindaas", "shandar", "lajawaab", "lajawab", "waah", "wah", "superb",
              "osm", "awsm", "dhansu", "badhiya", "maja", "mazza", "acha", "accha",
              "ekdum", "love", "pyaar", "shukriya", "dhanyawad"}
_NEG_SLANG = {"bakwas", "faltu", "bekar", "bekaar", "ghatiya", "wahiyat", "bura",
              "kharab", "boring", "ullu", "nafrat", "gussa", "naraaz"}


def _fast_path(text: str):
    stripped = text.strip().lower()

    if len(stripped) <= 2:
        return "Neutral", 0.6

    words = set(stripped.split())

    pos_hits = len(words & _POS_SLANG)
    neg_hits = len(words & _NEG_SLANG)

    if pos_hits > neg_hits and pos_hits >= 1:
        return "Positive", min(0.75 + 0.05 * pos_hits, 0.92)
    if neg_hits > pos_hits and neg_hits >= 1:
        return "Negative", min(0.75 + 0.05 * neg_hits, 0.92)

    return None


# ── Model inference ────────────────────────────────────────────────────────────
# Canonical label order used throughout the ensemble
_CANONICAL = ["Negative", "Neutral", "Positive"]

# Normalise a label string so casing/spacing differences don't matter — used in _align_probs


def _align_probs(probs: torch.Tensor, id2label: dict) -> torch.Tensor:
    """
    Reorder/collapse `probs` to always produce [Negative, Neutral, Positive].
    Handles both 3-class and 5-class (Very Negative/Negative/Neutral/Positive/Very Positive) models.
    """
    # 5-class: collapse Very Negative→Negative, Very Positive→Positive
    _5CLASS_MAP = {
        "very negative": 0, "negative": 0, "neg": 0,
        "neutral":       1, "neu": 1,
        "positive":      2, "pos": 2, "very positive": 2,
    }
    _3CLASS_MAP = {
        "negative": 0, "neg": 0,
        "neutral":  1, "neu": 1,
        "positive": 2, "pos": 2,
    }
    label_map = _5CLASS_MAP if len(id2label) == 5 else _3CLASS_MAP
    try:
        aligned = torch.zeros(3, device=probs.device)
        for native_idx, label in id2label.items():
            canonical_idx = label_map[label.lower()]
            aligned[canonical_idx] += probs[native_idx]
        return aligned
    except (KeyError, IndexError):
        print(f"[sentiment] WARNING: could not align labels {id2label}, using raw order")
        return probs[:3]


def _infer_aligned(tokenizer, model, text: str) -> torch.Tensor:
    """Run inference and return probs aligned to [Negative, Neutral, Positive]."""
    device = _device()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze()
    return _align_probs(probs, model.config.id2label)


# ── Ensemble ───────────────────────────────────────────────────────────────────
@lru_cache(maxsize=512)
def _ensemble(text):
    _load_models()

    p_muril = _infer_aligned(_muril_tokenizer, _muril_model, text)
    p_xlmr  = _infer_aligned(_xlmr_tokenizer,  _xlmr_model,  text)
    p_multi = _infer_aligned(_multi_tokenizer,  _multi_model,  text)

    probs = MURIL_WEIGHT * p_muril + XLMR_WEIGHT * p_xlmr + MULTI_WEIGHT * p_multi

    conf, idx = torch.max(probs, dim=0)

    return _CANONICAL[idx.item()], conf.item()


# ── Public API ─────────────────────────────────────────────────────────────────
def predict_sentiment(text: str):

    fast = _fast_path(text)
    if fast:
        return fast

    clean = _preprocess(text)

    if not clean:
        return "Neutral", 0.55

    label, conf = _ensemble(clean)

    boost = _emoji_score(text)
    conf = max(0, min(conf + boost, 1))

    return label, round(conf, 2)