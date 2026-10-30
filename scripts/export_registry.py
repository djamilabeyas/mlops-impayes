"""Exporte le registre de modèles MLflow (versions, alias, métriques, traçabilité) en JSON.

Usage : python scripts/export_registry.py [URL_MLFLOW] [FICHIER_SORTIE]
Par défaut : http://localhost:5000 -> registry/model_registry.json
Uniquement la bibliothèque standard (API REST de MLflow).
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

MODEL = "default-risk-model"


def api(base: str, path: str, **params) -> dict:
    url = f"{base}/api/2.0/mlflow/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def main(base: str, out: str) -> None:
    model = api(base, "registered-models/get", name=MODEL)["registered_model"]
    aliases = {a["alias"]: a["version"] for a in model.get("aliases", [])}
    versions = api(base, "model-versions/search", filter=f"name='{MODEL}'").get("model_versions", [])
    export = {"name": model["name"], "aliases": aliases, "versions": []}
    for v in sorted(versions, key=lambda x: int(x["version"])):
        run = api(base, "runs/get", run_id=v["run_id"])["run"]
        data = run["data"]
        export["versions"].append({
            "version": v["version"],
            "status": v.get("status"),
            "aliases": [a for a, ver in aliases.items() if ver == v["version"]],
            "run_id": v["run_id"],
            "metrics": {m["key"]: round(m["value"], 4) for m in data.get("metrics", [])},
            "params": {p["key"]: p["value"] for p in data.get("params", [])},
            "tags": {t["key"]: t["value"] for t in data.get("tags", []) if not t["key"].startswith("mlflow.")},
        })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(export, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:5000",
         sys.argv[2] if len(sys.argv) > 2 else "registry/model_registry.json")
