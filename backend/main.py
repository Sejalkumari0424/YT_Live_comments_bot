"""
backend/main.py
===============
FastAPI server exposing live-chat insights from Redis.

Start with:
    uvicorn backend.main:app --reload --port 8000

Endpoints
---------
GET /get_messages       Last 50 raw messages (full payload)
GET /sentiment_trend    Last 200 messages — time + sentiment + confidence
GET /sentiment_summary  Aggregate counts + avg confidence (last 200)
GET /topic_stats        Per-topic counts (last 100 messages)
GET /live_stats         Combined snapshot for dashboard widgets
GET /health             Redis connectivity check
"""

import json

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import REDIS_HOST, REDIS_PORT, REDIS_DB

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube Live Chat Insights API",
    description="Real-time Hinglish sentiment + topic analysis for live streams",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Redis ──────────────────────────────────────────────────────────────────────
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
    socket_connect_timeout=5,
)

VALID_TOPICS    = {"Appreciation", "Question", "Promo", "Spam", "General"}
VALID_SENTIMENT = {"Positive", "Neutral", "Negative"}


def _get_messages(n: int) -> list[dict]:
    """Fetch last n messages from Redis, safely parse each."""
    raws = r.lrange("chat_messages", -n, -1)
    out  = []
    for raw in raws:
        try:
            out.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    return out


def _redis_check():
    try:
        r.ping()
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check Redis connectivity."""
    _redis_check()
    return {"status": "ok", "redis": f"{REDIS_HOST}:{REDIS_PORT}"}


@app.get("/get_messages")
def get_messages(limit: int = 50):
    """
    Return the last `limit` chat messages with full payload.
    Each item: author, text, sentiment, confidence, topic, topic_conf, time.
    """
    _redis_check()
    return _get_messages(min(limit, 200))


@app.get("/sentiment_trend")
def sentiment_trend():
    """
    Time-series data for the last 200 messages.
    Useful for plotting sentiment over time in a frontend chart.
    Returns: [{time, sentiment, confidence}, ...]
    """
    _redis_check()
    msgs = _get_messages(200)
    return [
        {
            "time":       m.get("time"),
            "sentiment":  m.get("sentiment"),
            "confidence": m.get("confidence"),
        }
        for m in msgs
    ]


@app.get("/sentiment_summary")
def sentiment_summary():
    """
    Aggregate sentiment counts and average confidence for the last 200 messages.
    Returns: {counts: {Positive, Neutral, Negative}, avg_confidence, total_messages}
    """
    _redis_check()
    msgs   = _get_messages(200)
    counts = {s: 0 for s in VALID_SENTIMENT}
    total_conf = 0.0

    for m in msgs:
        s = m.get("sentiment", "Neutral")
        if s in counts:
            counts[s] += 1
        try:
            total_conf += float(m.get("confidence", 0))
        except (TypeError, ValueError):
            pass

    n = len(msgs)
    return {
        "counts":         counts,
        "avg_confidence": round(total_conf / n, 3) if n else 0.0,
        "total_messages": n,
    }


@app.get("/topic_stats")
def topic_stats():
    """
    Per-topic message counts for the last 100 messages.
    Returns: {Appreciation: int, Question: int, Promo: int, Spam: int, General: int}
    """
    _redis_check()
    msgs   = _get_messages(100)
    counts = {t: 0 for t in VALID_TOPICS}

    for m in msgs:
        label = m.get("topic", "General")
        if label not in VALID_TOPICS:
            label = "General"
        counts[label] += 1

    return counts


@app.get("/live_stats")
def live_stats():
    """
    Combined snapshot — sentiment summary + topic stats in one request.
    Use this for dashboard widgets to avoid two round-trips.
    """
    _redis_check()
    msgs = _get_messages(200)

    # Sentiment
    s_counts   = {s: 0 for s in VALID_SENTIMENT}
    total_conf = 0.0
    for m in msgs:
        s = m.get("sentiment", "Neutral")
        if s in s_counts:
            s_counts[s] += 1
        try:
            total_conf += float(m.get("confidence", 0))
        except (TypeError, ValueError):
            pass
    n = len(msgs)

    # Topics (last 100 only)
    t_counts = {t: 0 for t in VALID_TOPICS}
    for m in msgs[-100:]:
        label = m.get("topic", "General")
        if label not in VALID_TOPICS:
            label = "General"
        t_counts[label] += 1

    # Recent messages (last 10 for live feed widget)
    recent = msgs[-10:]

    return {
        "sentiment": {
            "counts":         s_counts,
            "avg_confidence": round(total_conf / n, 3) if n else 0.0,
            "total_messages": n,
        },
        "topics":  t_counts,
        "recent":  recent,
    }