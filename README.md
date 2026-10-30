# MLOps - Prédiction des impayés télécom (default_risk)

Dataset : Telecom Credit Scoring Data (Nigeria), Hugging Face
`electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria`
(à vérifier : le nom exact figure dans l'énoncé, ajuster `params.yaml > data.hf_dataset`).

## Démarrage rapide
```bash
pip install -r requirements.txt
git init && dvc init                  # une seule fois
dvc repro                             # exécute prepare + train et crée dvc.lock (ne pas faire dvc add : la donnée est déjà une sortie de dvc.yaml)
mlflow server --port 5000 &           # tracking + registry (optionnel : sinon SQLite local mlflow.db)
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.train                   # entraîne, évalue (gate), enregistre en staging
pytest -q                             # tests
docker build -t default-risk-api . && docker run -p 8000:8000 default-risk-api
docker compose up                     # API + MLflow + Prometheus + Grafana
# Swagger/OpenAPI : http://localhost:8000/docs
```

## Correspondance avec les livrables
| # | Livrable | Emplacement |
|---|---|---|
| 1 | Dépôt Git structuré | `src/`, `notebooks/`, `tests/`, `flows/`, `scripts/`, `monitoring/`, `registry/`, `docs/` |
| 2 | Données versionnées | `dvc.yaml`, `params.yaml`, prétraitements dans `src/train.py` |
| 3 | Modèle et métriques | `models/metrics.json`, `models/meta.json`, artefact `model` de chaque run GitHub Actions, runs MLflow |
| 4 | API de serving | `src/api.py` (`/predict`, `/health`, `/drift/check`, `/docs`) |
| 5 | Image Docker | `Dockerfile`, `docker-compose.yml` |
| 6 | Pipeline CI/CD/CT | `.github/workflows/ci-cd.yml`, `ct.yml`, `flows/retrain_flow.py` |
| 7 | Registre de modèles | MLflow Model Registry `default-risk-model` (alias `staging` / `production`), export `registry/model_registry.json`, artefact `mlflow-registry` de chaque run, captures `docs/captures/mlflow_*.png` |
| 8 | Dashboard monitoring | `monitoring/` (Prometheus, alertes, Grafana), captures `docs/captures/grafana_dashboard.png`, `docs/captures/prometheus_alertes.png` |
| 9 | Rapport | `docs/Rapport_MLOps_Impayes.pdf`, `docs/Rapport_MLOps_Impayes.docx` |

## Points à adapter après la première exécution sur les vraies données
- Vérifier que `default_risk` est binaire (0/1) et lister les 15 colonnes (`python -m src.data`).
- Retirer les colonnes identifiants ou fuyantes via `data.drop_columns`.
- Recalibrer la gate (`gate.min_f1`, `gate.min_auc`) selon les résultats réels.
- Promotion en `production` : manuelle (alias MLflow), après canari et validation.
