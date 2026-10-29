import numpy as np
import pandas as pd

def psi(expected, actual, bins: int = 10) -> float:
    e = pd.Series(expected).dropna().astype(float)
    a = pd.Series(actual).dropna().astype(float)
    if len(e) == 0 or len(a) == 0:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.clip(np.histogram(e, edges)[0] / len(e), 1e-6, None)
    pa = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))

def drift_report(ref: pd.DataFrame, cur: pd.DataFrame, threshold: float = 0.2) -> dict:
    cols = [c for c in ref.select_dtypes(include="number").columns if c in cur.columns]
    scores = {c: psi(ref[c], cur[c]) for c in cols}
    psi_max = max(scores.values()) if scores else 0.0
    drifted = [c for c, v in scores.items() if v > threshold]
    return {"psi": scores, "psi_max": psi_max, "drifted_features": drifted,
            "drift_detected": bool(drifted)}

def should_retrain(psi_max: float, f1_real, p: dict) -> dict:
    """Déclenche un entraînement CANDIDAT. La promotion reste soumise à la gate et à la validation humaine."""
    reasons = []
    if psi_max > p["monitoring"]["psi_threshold"]:
        reasons.append(f"data drift: PSI max {psi_max:.2f} > {p['monitoring']['psi_threshold']}")
    if f1_real is not None and f1_real < p["monitoring"]["f1_threshold"]:
        reasons.append(f"concept drift probable: F1 réel {f1_real:.2f} < {p['monitoring']['f1_threshold']}")
    return {"retrain": bool(reasons), "reasons": reasons}
