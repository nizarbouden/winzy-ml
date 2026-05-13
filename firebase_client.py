"""
firebase_client.py
──────────────────
Initialise Firebase Admin et expose une fonction pour lire les événements
utilisateur depuis Firestore (collection userEvents).
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

_db: Optional[object] = None


def get_db():
    global _db
    if _db is not None:
        return _db

    if firebase_admin._apps:
        _db = firestore.client()
        return _db

    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT", "")
    if not raw:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT env var is missing")

    try:
        sa = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"FIREBASE_SERVICE_ACCOUNT is not valid JSON: {e}")

    cred = credentials.Certificate(sa)
    firebase_admin.initialize_app(cred)
    _db = firestore.client()
    logger.info("✅ Firebase Admin initialised")
    return _db


# ─────────────────────────────────────────────────────────────────────────────
#  Poids par type d'événement
# ─────────────────────────────────────────────────────────────────────────────
EVENT_WEIGHTS: dict[str, float] = {
    "view":     1.0,
    "click":    2.0,
    "like":     3.0,
    "search":   1.5,
    "purchase": 5.0,
}


def fetch_events(days: int = 30) -> list[dict]:
    """
    Lit tous les userEvents des `days` derniers jours.
    Retourne une liste de dicts :
      { userId, productId, eventType, category, price, score, timestamp }
    """
    db = get_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    snap = (
        db.collection("userEvents")
        .where(filter=firestore.FieldFilter("timestamp", ">=", cutoff))
        .order_by("timestamp")
        .get()
    )

    events = []
    for doc in snap:
        d = doc.to_dict()
        event_type = d.get("eventType", "view")
        score = EVENT_WEIGHTS.get(event_type, 1.0)
        events.append(
            {
                "userId":    d.get("userId", ""),
                "productId": d.get("productId", ""),
                "eventType": event_type,
                "category":  d.get("category", ""),
                "price":     float(d.get("price", 0)),
                "score":     score,
                "timestamp": d.get("timestamp"),
            }
        )

    logger.info(f"📦 {len(events)} events fetched (last {days} days)")
    return events


def fetch_products(limit: int = 2000) -> list[dict]:
    """
    Lit les produits approuvés pour le content-based filtering.
    Retourne: { productId, category, price, sellerId }
    """
    db = get_db()
    snap = (
        db.collection("products")
        .where(filter=firestore.FieldFilter("status", "==", "approved"))
        .limit(limit)
        .get()
    )

    products = []
    for doc in snap:
        d = doc.to_dict()
        products.append(
            {
                "productId": doc.id,
                "category":  d.get("category", ""),
                "price":     float(d.get("price", 0)),
                "sellerId":  d.get("sellerId", ""),
            }
        )

    logger.info(f"📦 {len(products)} products fetched")
    return products


def fetch_popular_products(days: int = 7, limit: int = 20) -> list[str]:
    """
    Retourne les IDs des produits les plus populaires (cold-start).
    Basé sur le nombre d'événements likes + purchases récents.
    """
    events = fetch_events(days=days)
    scores: dict[str, float] = {}
    for e in events:
        if e["eventType"] in ("like", "purchase", "click"):
            pid = e["productId"]
            scores[pid] = scores.get(pid, 0) + e["score"]

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return sorted_ids[:limit]
