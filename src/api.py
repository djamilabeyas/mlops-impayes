import json, os, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from src.config import load_params
from src.monitoring.drift import drift_report

REQS = Counter("http_requests_total", "Requêtes HTTP", ["status"])
LAT = Histogram("request_latency_seconds", "Latence des requêtes",
                buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2))
PSI_G = Gauge("data_drift_psi_max", "PSI maximal sur les variables numériques")
state: Dict[str, Any] = {"model": None, "meta": {}, "clients": None, "clients_path": None}
STATIC = Path(__file__).parent / "static"
# Colonnes jamais exposées par l'API : donnée personnelle, date technique, étiquette historique
HIDDEN = ["phone_number", "assessment_date", "default_risk"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    if path.exists():
        state["model"] = joblib.load(path)
        meta = path.parent / "meta.json"
        state["meta"] = json.loads(meta.read_text()) if meta.exists() else {}
    yield

app = FastAPI(title="API de prédiction des impayés télécom", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())

@app.middleware("http")
async def track(request: Request, call_next):
    if request.url.path.startswith("/metrics"):
        return await call_next(request)
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
        REQS.labels(status=str(resp.status_code)).inc()
        return resp
    except Exception:
        REQS.labels(status="500").inc()
        raise
    finally:
        LAT.observe(time.perf_counter() - t0)

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., examples=[{"tenure_months": 24, "late_payments": 2}])

class PredictResponse(BaseModel):
    default_risk_score: float
    prediction: int
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "model_version": state["meta"].get("version", "n/a")}

def score_features(features: Dict[str, Any]) -> float:
    """Calcule le score de risque et journalise la prédiction (alimente le suivi de dérive)."""
    if state["model"] is None:
        raise HTTPException(503, "Modèle non chargé")
    cols = state["meta"].get("features") or list(features)
    X = pd.DataFrame([features]).reindex(columns=cols)
    score = float(state["model"].predict_proba(X)[0, 1])
    log = Path(os.getenv("PREDICTION_LOG", "prediction_log.jsonl"))
    with log.open("a") as f:
        f.write(json.dumps({"ts": time.time(), **features, "score": score}, default=str) + "\n")
    return score

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    score = score_features(req.features)
    return PredictResponse(default_risk_score=score, prediction=int(score >= 0.5),
                           model_version=str(state["meta"].get("version", "n/a")))

def clients() -> pd.DataFrame:
    """Fichier clients chargé une seule fois, indexé par customer_id (sans les colonnes masquées)."""
    path = Path(os.getenv("CLIENTS_PATH", "data/raw/telecom.csv"))
    if state["clients"] is None or state["clients_path"] != path:
        if not path.exists():
            raise HTTPException(503, "Fichier clients indisponible")
        df = pd.read_csv(path).drop(columns=HIDDEN, errors="ignore")
        # le jeu synthétique contient des identifiants en double : on garde la première fiche
        df = df.drop_duplicates("customer_id", keep="first").set_index("customer_id")
        state["clients"], state["clients_path"] = df, path
    return state["clients"]

@app.get("/clients")
def search_clients(q: str = "", limit: int = 8):
    """Suggestions d'identifiants commençant par q (saisie semi-automatique de l'interface)."""
    q = q.strip().upper()
    if len(q) < 3:
        return []
    idx = clients().index
    return idx[idx.str.startswith(q)][:max(1, min(limit, 20))].tolist()

@app.get("/clients/{customer_id}")
def score_client(customer_id: str):
    """Recherche un client par son identifiant et renvoie son score de risque."""
    df, cid = clients(), customer_id.strip().upper()
    if cid not in df.index:
        raise HTTPException(404, f"Client {cid} introuvable")
    row = df.loc[cid]
    profile = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    score = score_features(profile)
    return {"customer_id": cid, "default_risk_score": score, "prediction": int(score >= 0.5),
            "risk": "élevé" if score >= 0.5 else "faible",
            "model_version": str(state["meta"].get("version", "n/a")), "profile": profile}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")

@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")

@app.post("/drift/check")
def drift_check(last_n: int = 1000):
    log = Path(os.getenv("PREDICTION_LOG", "prediction_log.jsonl"))
    ref_path = Path(os.getenv("MODEL_PATH", "models/model.joblib")).parent / "reference.csv"
    if not log.exists() or not ref_path.exists():
        raise HTTPException(404, "Pas de journal de prédictions ou de référence")
    lines = log.read_text().splitlines()[-last_n:]
    cur = pd.DataFrame([json.loads(x) for x in lines])
    rep = drift_report(pd.read_csv(ref_path), cur, load_params()["monitoring"]["psi_threshold"])
    PSI_G.set(rep["psi_max"])
    return rep
