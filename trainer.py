"""
trainer.py
──────────
Construit les matrices de similarité depuis les événements et produits bruts.
Retourne un objet TrainedModel immuable — pas d'état global ici.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """Snapshot immutable du modèle entraîné."""

    # ── Collaborative Filtering ──────────────────────────────────────────────
    user_item_matrix: Optional[pd.DataFrame] = None   # shape: (users, products)
    user_sim_matrix:  Optional[np.ndarray]   = None   # shape: (users, users)
    user_index:       dict[str, int]         = field(default_factory=dict)

    # ── Content-Based Filtering ──────────────────────────────────────────────
    product_features:   Optional[pd.DataFrame] = None  # shape: (products, features)
    product_sim_matrix: Optional[np.ndarray]   = None  # shape: (products, products)

    # ── Index commun produit ─────────────────────────────────────────────────
    product_index:  dict[str, int] = field(default_factory=dict)

    # ── Cold-start ───────────────────────────────────────────────────────────
    popular_ids: list[str] = field(default_factory=list)

    # ── Flags ────────────────────────────────────────────────────────────────
    collab_ok:  bool = False
    content_ok: bool = False

    @property
    def is_ready(self) -> bool:
        return self.collab_ok or self.content_ok


def build_model(
    events:      list[dict],
    products:    list[dict],
    popular_ids: list[str],
) -> TrainedModel:
    """
    Entraîne le modèle depuis les données brutes.
    Appelé en arrière-plan par main.py ; retourne un TrainedModel propre.
    """
    logger.info(f"🧠 build_model — {len(events)} events | {len(products)} products")

    model = TrainedModel(popular_ids=list(popular_ids))

    # ── 1. Collaborative Filtering ───────────────────────────────────────────
    if events:
        model = _build_collab(events, model)
    else:
        logger.warning("⚠️  Pas d'événements — collaborative filtering ignoré")

    # ── 2. Content-Based Filtering ───────────────────────────────────────────
    if products:
        model = _build_content(products, model)
    else:
        logger.warning("⚠️  Pas de produits — content-based filtering ignoré")

    logger.info(
        f"✅ build_model terminé — "
        f"users:{len(model.user_index)} "
        f"products:{len(model.product_index)} "
        f"collab:{model.collab_ok} content:{model.content_ok}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 1 — Matrice user × product (collaborative)
# ─────────────────────────────────────────────────────────────────────────────
def _build_collab(events: list[dict], model: TrainedModel) -> TrainedModel:
    df = pd.DataFrame(events)
    df = df[df["userId"].str.strip() != ""]
    df = df[df["productId"].str.strip() != ""]

    if df.empty:
        logger.warning("⚠️  Tous les événements sont vides après nettoyage")
        return model

    # Somme des scores par (user, product)
    agg = df.groupby(["userId", "productId"])["score"].sum().reset_index()

    pivot = agg.pivot(index="userId", columns="productId", values="score").fillna(0)

    # Normalise chaque ligne pour éviter le biais des users très actifs
    mat = pivot.values.astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mat_norm = mat / norms

    user_sim = cosine_similarity(mat_norm).astype(np.float32)

    model.user_item_matrix = pivot
    model.user_sim_matrix  = user_sim
    model.user_index       = {uid: i for i, uid in enumerate(pivot.index)}
    model.product_index    = {pid: j for j, pid in enumerate(pivot.columns)}
    model.collab_ok        = True
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  ÉTAPE 2 — Matrice produit × features (content-based)
# ─────────────────────────────────────────────────────────────────────────────
def _build_content(products: list[dict], model: TrainedModel) -> TrainedModel:
    pdf = pd.DataFrame(products)
    pdf = pdf[pdf["productId"].str.strip() != ""]
    pdf = pdf.drop_duplicates("productId").reset_index(drop=True)

    if pdf.empty:
        logger.warning("⚠️  Tous les produits sont vides après nettoyage")
        return model

    # Catégorie → one-hot
    cat_dummies = pd.get_dummies(pdf["category"], prefix="cat")

    # Prix → normalisé 0-1
    scaler = MinMaxScaler()
    price_norm = scaler.fit_transform(pdf[["price"]].fillna(0))

    features = np.hstack([
        cat_dummies.values.astype(np.float32),
        price_norm.astype(np.float32),
    ])

    prod_sim = cosine_similarity(features).astype(np.float32)
    prod_idx = {pid: i for i, pid in enumerate(pdf["productId"])}

    model.product_features   = pdf
    model.product_sim_matrix = prod_sim
    model.content_ok         = True

    # Fusionne avec l'index collab (qui peut couvrir moins de produits)
    for pid, idx in prod_idx.items():
        if pid not in model.product_index:
            model.product_index[pid] = idx

    return model
