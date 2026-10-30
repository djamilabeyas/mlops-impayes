"""Envoie des clients réels à l'API pour alimenter le monitoring (Prometheus / Grafana).

Usage : python scripts/generate_traffic.py [URL_API] [NB_REQUETES] [FICHIER_CSV]
Uniquement la bibliothèque standard.
"""
import csv
import json
import sys
import urllib.request

DROP = {"customer_id", "phone_number", "assessment_date", "default_risk"}


def typed(value: str):
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def main(base: str, n: int, path: str) -> None:
    sent = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            client = {k: typed(v) for k, v in row.items() if k not in DROP}
            req = urllib.request.Request(f"{base}/predict", data=json.dumps({"features": client}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
            sent += 1
            if sent >= n:
                break
    with urllib.request.urlopen(urllib.request.Request(f"{base}/drift/check", method="POST"), timeout=30) as r:
        drift = json.loads(r.read())
    print(f"{sent} prédictions envoyées ; PSI max = {drift['psi_max']:.3f} ; dérive détectée : {drift['drift_detected']}")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000",
         int(sys.argv[2]) if len(sys.argv) > 2 else 500,
         sys.argv[3] if len(sys.argv) > 3 else "data/raw/telecom.csv")
