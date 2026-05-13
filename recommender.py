"""
recommender.py
──────────────
Logique d'inférence uniquement — utilise un TrainedModel déjà construit.
L'entraînement (construction des matrices) est dans trainer.py.

Algorithme hybride :
  • 60 % Collaborative Filtering  (user-user cosine similarity)
  • 40 % Content-Based Filtering  (product-feature cosine similarity)
  • Cold-start : produits populaires si l'utilisateur n'a pas d'historique
"""

import logging
import threading
from typing import Optional

import numpy as np

from trainer import TrainedModel

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  État global — remplacé atomiquement à chaque ré-entraînement
# ─────────────────────────────────────────────────────────────────────────────
_lock:  threading.Lock = threading.Lock()
_model: Optional[TrainedModel] = None


def load_model(model: TrainedModel) -> None:
    """Remplace le modèle en mémoire de façon thread-safe."""
    global _model
    with _lock:
        _model = model
    logger.info("✅ [Recommender] Modèle chargé en mémoire")


def is_ready() -> bool:
    with _lock:
        return _model is not None and _model.is_ready


# ─────────────────────────────────────────────────────────────────────────────
#  Poids hybride
# ─────────────────────────────────────────────────────────────────────────────
_COLLAB_W  = 0.60
_CONTENT_W = 0.40
_TOP_USERS = 10  # nombre de voisins utilisés pour le collaborative filtering


# ─────────────────────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE PUBLIC
# ─────────────────────────────────────────────────────────────────────────────
def recommend(
    user_id:          str,
    limit:            int = 20,
    exclude_seller_id: str = "",
) -> list[str]:
    """
    Retourne une liste ordonnée de productIds recommandés pour user_id.
    Dégradation :
      • Pas de collab  → 100 % content-based
      • Pas de content → 100 % collaborative
      • Rien du tout   → produits populaires (cold-start)
    """
    with _lock:
        model = _model

    if model is None or not model.is_ready:
        return []

    has_collab  = model.collab_ok  and user_id in model.user_index
    has_content = model.content_ok and model.user_item_matrix is not None

    if not has_collab and not has_content:
        logger.debug(f"❄️  Cold-start pour {user_id}")
        return _cold_start(model, user_id, exclude_seller_id, limit)

    seen   = _seen_products(model, user_id)
    scores: dict[str, float] = {}

    if has_collab:
        for pid, s in _collab_scores(model, user_id, seen).items():
            scores[pid] = scores.get(pid, 0.0) + _COLLAB_W * s

    if has_content and seen:
        for pid, s in _content_scores(model, seen).items():
            scores[pid] = scores.get(pid, 0.0) + _CONTENT_W * s

    scores = _exclude(model, scores, seen, exclude_seller_id)

    if not scores:
        return _cold_start(model, user_id, exclude_seller_id, limit)

    return sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS PRIVÉS
# ─────────────────────────────────────────────────────────────────────────────

def _seen_products(model: TrainedModel, user_id: str) -> set[str]:
    """Produits avec lesquels l'utilisateur a interagi."""
    if model.user_item_matrix is None or user_id not in model.user_index:
        return set()
    row = model.user_item_matrix.loc[user_id]
    return set(row[row > 0].index.tolist())


def _collab_scores(
    model: TrainedModel,
    user_id: str,
    seen: set[str],
) -> dict[str, float]:
    u_idx   = model.user_index[user_id]
    sim_row = model.user_sim_matrix[u_idx].copy()
    sim_row[u_idx] = -1  # exclut soi-même

    top_indices = np.argsort(sim_row)[::-1][:_TOP_USERS]

    mat  = model.user_item_matrix.values
    cols = list(model.user_item_matrix.columns)

    scores: dict[str, float] = {}
    for idx in top_indices:
        similarity = float(model.user_sim_matrix[u_idx, idx])
        if similarity <= 0:
            continue
        for j, pid in enumerate(cols):
            if pid in seen:
                continue
            val = float(mat[idx, j])
            if val > 0:
                scores[pid] = scores.get(pid, 0.0) + similarity * val

    return _normalize(scores)


def _content_scores(
    model: TrainedModel,
    seen: set[str],
) -> dict[str, float]:
    if model.product_sim_matrix is None or model.product_features is None:
        return {}

    all_pids = list(model.product_features["productId"])
    scores: dict[str, float] = {}

    for seen_pid in seen:
        if seen_pid not in model.product_index:
            continue
        row_idx = model.product_index[seen_pid]
        sim_row = model.product_sim_matrix[row_idx]

        for j, pid in enumerate(all_pids):
            if pid in seen:
                continue
            scores[pid] = scores.get(pid, 0.0) + float(sim_row[j])

    return _normalize(scores)


def _exclude(
    model: TrainedModel,
    scores: dict[str, float],
    seen: set[str],
    exclude_seller_id: str,
) -> dict[str, float]:
    seller_products: set[str] = set()
    if exclude_seller_id and model.product_features is not None:
        mask = model.product_features["sellerId"] == exclude_seller_id
        seller_products = set(model.product_features.loc[mask, "productId"])

    return {
        pid: s
        for pid, s in scores.items()
        if pid not in seen and pid not in seller_products
    }


def _cold_start(
    model: TrainedModel,
    user_id: str,
    exclude_seller_id: str,
    limit: int,
) -> list[str]:
    seen = _seen_products(model, user_id)

    seller_products: set[str] = set()
    if exclude_seller_id and model.product_features is not None:
        mask = model.product_features["sellerId"] == exclude_seller_id
        seller_products = set(model.product_features.loc[mask, "productId"])

    return [
        pid for pid in model.popular_ids
        if pid not in seen and pid not in seller_products
    ][:limit]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return scores
    max_s = max(scores.values())
    if max_s <= 0:
        return scores
    return {k: v / max_s for k, v in scores.items()}
