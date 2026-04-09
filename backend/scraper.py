"""
backend/scraper.py
==================
Fetches live YouTube chat comments, runs sentiment + topic classification,
and pushes results to Redis.

Run this as a standalone process:
    python -m backend.scraper

or directly:
    python backend/scraper.py
"""

import json
import logging
import time
from datetime import datetime

import pytchat
import redis

from backend.config import (
    VIDEO_ID,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
)
from ml.sentiment_model import predict_sentiment
from ml.topic_model import predict_topic, VALID_TOPICS

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")

# ── Redis connection ───────────────────────────────────────────────────────────
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
    socket_connect_timeout=5,
)

try:
    r.ping()
    logger.info("Redis connected ✓")
except redis.ConnectionError as e:
    logger.error("Cannot connect to Redis: %s", e)
    raise SystemExit(1)

MAX_REDIS_MESSAGES = 1000  # cap the Redis list size


# ── Helpers ────────────────────────────────────────────────────────────────────
def _safe_sentiment(text: str) -> tuple[str, float]:
    """Run sentiment prediction with fallback on any error."""
    try:
        return predict_sentiment(text)
    except Exception as exc:
        logger.error("predict_sentiment failed for %r: %s", text[:60], exc)
        return "Neutral", 0.50


def _safe_topic(text: str) -> tuple[str, float]:
    """Run topic prediction with fallback on any error."""
    try:
        topic, conf = predict_topic(text)
        if topic not in VALID_TOPICS:
            logger.warning("Invalid topic %r — using 'General'", topic)
            return "General", 0.50
        return topic, conf
    except Exception as exc:
        logger.error("predict_topic failed for %r: %s", text[:60], exc)
        return "General", 0.50


def _push_to_redis(data: dict) -> None:
    """Push message to Redis list and trim to MAX_REDIS_MESSAGES."""
    pipe = r.pipeline()
    pipe.rpush("chat_messages", json.dumps(data))
    pipe.ltrim("chat_messages", -MAX_REDIS_MESSAGES, -1)
    pipe.execute()


# ── Main loop ──────────────────────────────────────────────────────────────────
def run() -> None:
    logger.info("Starting live chat scraper for video: %s", VIDEO_ID)

    chat = pytchat.create(video_id=VIDEO_ID)
    if not chat.is_alive():
        logger.error("Could not connect to live chat. Is the stream live?")
        return

    logger.info("Live chat connected ✓  — press Ctrl+C to stop")

    while chat.is_alive():
        try:
            for c in chat.get().sync_items():
                text   = c.message.strip()
                author = c.author.name

                if not text:
                    continue

                # ── Classify ──────────────────────────────────────────────
                sentiment, s_conf = _safe_sentiment(text)
                topic,     t_conf = _safe_topic(text)

                # ── Build payload ─────────────────────────────────────────
                message_data = {
                    "author":      author,
                    "text":        text,
                    "sentiment":   sentiment,
                    "confidence":  round(s_conf, 3),
                    "topic":       topic,
                    "topic_conf":  round(t_conf, 3),
                    "time":        datetime.now().isoformat(),
                }

                # ── Store ─────────────────────────────────────────────────
                _push_to_redis(message_data)

                logger.info(
                    "[%s] %s | sentiment=%s(%.2f) topic=%s(%.2f) | %r",
                    message_data["time"][11:19],
                    author[:20],
                    sentiment, s_conf,
                    topic, t_conf,
                    text[:60],
                )

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as exc:
            logger.error("Unexpected error in chat loop: %s", exc, exc_info=True)

        time.sleep(1)

    logger.info("Chat stream ended.")


if __name__ == "__main__":
    run()