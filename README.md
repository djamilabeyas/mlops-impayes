# MLOps – Prédiction des impayés télécom (`default_risk`)

Industrialisation d'un modèle de classification des clients télécom à risque d'impayé : versioning des données, entraînement suivi, registre de modèles, API conteneurisée, chaîne CI/CD/CT et monitoring.

**Données** : Telecom Credit Scoring Data (Nigeria), Hugging Face `electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria` – 500 000 lignes, 15 colonnes. Cible : `default_risk` (`high` = impayé, `low` / `medium` = paie).

**Résultats** (jeu de test, 100 000 clients) : précision 0,496 · rappel 0,697 · F1 0,580 · AUC 0,746.

## Lancer le projet (Docker)

Prérequis : Git, Docker Desktop démarré, Python 3 (uniquement pour le script de trafic).

```bash
git clone https://github.com/djamilabeyas/mlops-impayes.git
cd mlops-impayes
docker compose run --rm train      # télécharge les données (Hugging Face), entraîne, enregistre dans MLflow
docker compose up -d --build       # API, MLflow, Prometheus, Grafana
python scripts/generate_traffic.py # trafic de démonstration vers l'API (alimente Grafana)
```

| Service | Adresse |
|---|---|
| API (documentation OpenAPI) | http://localhost:8000/docs |
| MLflow (expériences et registre) | http://localhost:5000 |
| Grafana (dashboard) | http://localhost:3000 |
| Prometheus (alertes) | http://localhost:9090/alerts |

Arrêt : `docker compose down`. Après un nouvel entraînement : `docker compose up -d --build api`.

## Développement local (sans Docker)

```bash
pip install -r requirements.txt
dvc repro                          # prepare (ingestion + contrôles) puis train (entraînement, gate, MLflow)
pytest -q                          # tests
python scripts/export_registry.py  # export du registre MLflow -> registry/model_registry.json
```

## Structure

```
src/                 data.py (ingestion, contrôles), train.py (entraînement, évaluation, MLflow), api.py (FastAPI), monitoring/drift.py (PSI)
notebooks/           01_exploration_entrainement.ipynb (exploration, entraînement, évaluation, dérive)
tests/               tests unitaires (entraînement, API, encodage de la cible, PSI, réentraînement)
flows/               retrain_flow.py (Prefect : contrôle de dérive et réentraînement)
scripts/             smoke_test.py, generate_traffic.py, export_registry.py
Dockerfile           image de l'API ; Dockerfile.train : image d'entraînement et de MLflow
monitoring/          prometheus.yml, alert_rules.yml, dashboard Grafana
registry/            model_registry.json (export du registre MLflow)
models/              metrics.json, meta.json
docs/                rapport (PDF, DOCX) et captures
.github/workflows/   ci-cd.yml (CI/CD), ct.yml (entraînement continu)
dvc.yaml, dvc.lock   pipeline de données versionné
params.yaml          paramètres (données, modèle, gate, seuils de monitoring)
```

## Correspondance avec les livrables

| # | Livrable | Emplacement |
|---|---|---|
| 1 | Dépôt Git structuré | `src/`, `notebooks/`, `tests/`, `flows/`, `scripts/`, `monitoring/`, `registry/`, `docs/` |
| 2 | Données versionnées | `dvc.yaml`, `dvc.lock` (empreintes des données, du code et des paramètres), `params.yaml` ; prétraitements décrits dans `notebooks/01_exploration_entrainement.ipynb` (section 4) et le rapport |
| 3 | Modèle et métriques | `models/metrics.json`, `models/meta.json`, notebook (matrice de confusion, ROC), artefact `model` de chaque run GitHub Actions, runs MLflow |
| 4 | API de serving | `src/api.py` (`/predict`, `/health`, `/metrics`, `/drift/check`, `/docs`) |
| 5 | Image Docker | `Dockerfile`, `docker-compose.yml`, image `ghcr.io/djamilabeyas/mlops-impayes` |
| 6 | Pipeline CI/CD/CT | `.github/workflows/ci-cd.yml`, `.github/workflows/ct.yml`, `flows/retrain_flow.py` |
| 7 | Registre de modèles | MLflow Model Registry `default-risk-model` (alias `staging` / `production`), `registry/model_registry.json`, artefact `mlflow-registry` de chaque run, captures `docs/captures/mlflow_*.png` |
| 8 | Dashboard monitoring | `monitoring/` (Prometheus, alertes, Grafana), captures `docs/captures/grafana_dashboard.png`, `docs/captures/prometheus_alertes.png` |
| 9 | Rapport | `docs/Rapport_MLOps_Impayes.pdf`, `docs/Rapport_MLOps_Impayes.docx` |
