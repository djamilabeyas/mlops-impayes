"""Test de fumée après déploiement : vérifie que l'API déployée répond correctement.

Usage : python scripts/smoke_test.py http://localhost:8000 models/reference.csv
Uniquement la bibliothèque standard, pour tourner sur n'importe quel serveur.
"""
import csv
import json
import sys
import time
import urllib.request


def get(url: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())


def typed(value: str):
    """Les valeurs du CSV sont du texte : on rend leur type (nombre, booléen) avant l'envoi."""
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def main(base: str, reference_csv: str) -> None:
    for _ in range(30):  # l'API peut mettre quelques secondes à démarrer
        try:
            status, body = get(f"{base}/health")
            if status == 200 and json.loads(body).get("model_loaded"):
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        sys.exit("ECHEC : /health ne répond pas ou le modèle n'est pas chargé")
    print("OK /health :", body.decode())

    with open(reference_csv, newline="") as f:
        client = {k: typed(v) for k, v in next(csv.DictReader(f)).items()}
    status, pred = post(f"{base}/predict", {"features": client})
    assert status == 200 and 0 <= pred["default_risk_score"] <= 1, pred
    print("OK /predict :", pred)

    for path in ("/openapi.json", "/docs", "/metrics"):
        status, _ = get(f"{base}{path}")
        assert status == 200, f"{path} -> {status}"
        print(f"OK {path}")
    print("Déploiement validé.")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2])
