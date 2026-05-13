"""
main.py
───────
Winzy ML — Service de recommandation (FastAPI)

Routes :
  GET  /                         health check
  POST /train                    (re)entraîne le modèle depuis Firestore
  GET  /recommend/{userId}       retourne les recommandations personnalisées
"""

import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

import firebase_client as fc
import recommender
from trainer import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CRON_SECRET = os.getenv("CRON_SECRET", "")


# ─────────────────────────────────────────────────────────────────────────────
#  Entraînement initial au démarrage (non bloquant)
# ─────────────────────────────────────────────────────────────────────────────
def _run_training():
    """Lit les données Firestore, construit le modèle, le charge en mémoire."""
    try:
        logger.info("🚀 Entraînement initial au démarrage���")
        events   = fc.fetch_events(days=30)
        products = fc.fetch_products(limit=2000)
        popular  = fc.fetch_popular_products(days=7, limit=20)
        model    = build_model(events, products, popular)
        recommender.load_model(model)
    except Exception as e:
        logger.error(f"❌ Entraînement initial échoué: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_run_training, daemon=True).start()
    yield


# ─────────────────────────────────────────────────────────────────────────────
#  App FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Winzy ML", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_secret(x_cron_secret: str) -> None:
    if CRON_SECRET and x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Non autorisé — CRON_SECRET invalide")


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status":    "Winzy ML running ✅",
        "ready":     recommender.is_ready(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/train")
def train(x_cron_secret: str = Header(default="")):
    """
    (Re)entraîne le modèle depuis Firestore.
    Appelé par le cron Express toutes les 6 heures.
    Exécuté en arrière-plan pour répondre immédiatement.
    """
    _check_secret(x_cron_secret)

    def _bg():
        try:
            events   = fc.fetch_events(days=30)
            products = fc.fetch_products(limit=2000)
            popular  = fc.fetch_popular_products(days=7, limit=20)
            model    = build_model(events, products, popular)
            recommender.load_model(model)
            logger.info(
                f"✅ Re-training OK — users:{len(model.user_index)} "
                f"products:{len(model.product_index)}"
            )
        except Exception as e:
            logger.error(f"❌ Re-training échoué: {e}")

    threading.Thread(target=_bg, daemon=True).start()
    return {"ok": True, "message": "Entraînement démarré en arrière-plan"}


@app.get("/recommend/{user_id}")
def get_recommendations(
    user_id:        str,
    limit:          int   = Query(default=20, ge=1, le=50),
    exclude_seller: str   = Query(default=""),
    x_cron_secret:  str   = Header(default=""),
):
    """
    Retourne une liste ordonnée de productIds recommandés pour user_id.
      - limit          : nombre de résultats (1–50, défaut 20)
      - exclude_seller : exclut les produits de ce sellerId
    """
    _check_secret(x_cron_secret)

    if not recommender.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Modèle pas encore prêt — réessayez dans quelques secondes",
        )

    try:
        ids = recommender.recommend(
            user_id,
            limit=limit,
            exclude_seller_id=exclude_seller,
        )
        return {"userId": user_id, "products": ids, "count": len(ids)}
    except Exception as e:
        logger.error(f"❌ /recommend/{user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
