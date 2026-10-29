import json, subprocess, sys
from typing import Optional
import pandas as pd
from prefect import flow, task
from src.config import load_params
from src.monitoring.drift import drift_report, should_retrain

@task
def check_drift(log_path: str = "prediction_log.jsonl") -> float:
    ref = pd.read_csv("models/reference.csv")
    cur = pd.read_json(log_path, lines=True).tail(1000)
    rep = drift_report(ref, cur, load_params()["monitoring"]["psi_threshold"])
    print(json.dumps(rep["drifted_features"]))
    return rep["psi_max"]

@task
def retrain_candidate() -> int:
    return subprocess.call([sys.executable, "-m", "src.train"])

@flow(name="monitoring-and-ct")
def monitoring_flow(f1_real: Optional[float] = None):
    p = load_params()
    decision = should_retrain(check_drift(), f1_real, p)
    print(decision)
    if decision["retrain"]:
        code = retrain_candidate()
        print("Candidat en staging, validation humaine requise" if code == 0
              else "Gate échouée: modèle de production conservé")

if __name__ == "__main__":
    monitoring_flow.serve(name="daily-drift-check", cron="0 2 * * *")
