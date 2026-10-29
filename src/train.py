import hashlib, json, os, subprocess, sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.config import load_params
from src.data import load_data

MODEL_NAME = "default-risk-model"

def build_pipeline(X: pd.DataFrame, p: dict):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=p["train"]["max_iter"], learning_rate=p["train"]["learning_rate"],
        class_weight="balanced", random_state=p["data"]["seed"])
    return Pipeline([("pre", pre), ("clf", clf)]), num, cat

def _git_commit() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:8]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]

def evaluate(pipe, X, y) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, proba)),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}

def train(df: pd.DataFrame, p: dict, track: bool = True, out_dir: str = "models") -> dict:
    target = p["data"]["target"]
    df = df.drop(columns=p["data"].get("drop_columns", []))
    X, y = df.drop(columns=[target]), df[target].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=p["data"]["test_size"], stratify=y, random_state=p["data"]["seed"])
    pipe, num, cat = build_pipeline(X, p)
    pipe.fit(Xtr, ytr)
    metrics = evaluate(pipe, Xte, yte)
    passed = metrics["f1"] >= p["gate"]["min_f1"] and metrics["auc"] >= p["gate"]["min_auc"]
    meta = {"features": X.columns.tolist(), "numeric": num, "categorical": cat,
            "metrics": metrics, "gate_passed": passed, "commit": _git_commit(),
            "data_hash": _data_hash(df), "version": "local"}

    if track:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
        mlflow.set_experiment("default-risk")
        with mlflow.start_run():
            mlflow.log_params({**p["train"], "seed": p["data"]["seed"]})
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})
            mlflow.log_dict({"confusion_matrix": metrics["confusion_matrix"]}, "confusion_matrix.json")
            mlflow.set_tags({"git_commit": meta["commit"], "data_hash": meta["data_hash"],
                             "gate_passed": str(passed)})
            if passed:
                info = mlflow.sklearn.log_model(pipe, "model", registered_model_name=MODEL_NAME)
                version = getattr(info, "registered_model_version", None)
                if version:
                    MlflowClient().set_registered_model_alias(MODEL_NAME, "staging", str(version))
                    meta["version"] = str(version)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if passed:
        joblib.dump(pipe, out / "model.joblib")
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        Xtr.sample(min(5000, len(Xtr)), random_state=p["data"]["seed"]).to_csv(out / "reference.csv", index=False)
    return {"metrics": metrics, "passed": passed, "meta": meta}

if __name__ == "__main__":
    params = load_params()
    res = train(load_data(params), params)
    print(json.dumps(res["metrics"], indent=2))
    if not res["passed"]:
        print("GATE ECHOUEE: modèle rejeté, la production n'est pas modifiée.")
        sys.exit(1)
