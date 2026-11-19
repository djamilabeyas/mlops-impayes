import pandas as pd
from fastapi.testclient import TestClient
from src.train import train

def test_train_and_serve(df, params, tmp_path, monkeypatch):
    res = train(df, params, track=False, out_dir=str(tmp_path))
    assert res["passed"] and res["metrics"]["auc"] > 0.7
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.jsonl"))
    from src.api import app
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is True
        row = df.drop(columns=["default_risk"]).iloc[0].to_dict()
        for _ in range(30):
            r = c.post("/predict", json={"features": row})
        assert r.status_code == 200 and 0 <= r.json()["default_risk_score"] <= 1
        assert c.post("/drift/check").status_code == 200


def test_cible_texte(df, params, tmp_path):
    d = df.copy()
    d["default_risk"] = d["default_risk"].map({1: "high", 0: "low"})
    d["customer_id"] = [f"CUST{i}" for i in range(len(d))]
    res = train(d, params, track=False, out_dir=str(tmp_path))
    assert "customer_id" not in res["meta"]["features"]
    assert res["metrics"]["auc"] > 0.7


def test_commit_sans_git(monkeypatch):
    import subprocess
    from src import train as t

    def absent(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(subprocess, "check_output", absent)
    assert len(t._git_commit()) == 7


def test_interface_et_recherche_client(df, params, tmp_path, monkeypatch):
    train(df, params, track=False, out_dir=str(tmp_path))
    d = df.drop(columns=["default_risk"]).copy()
    d.insert(0, "customer_id", [f"CUST{i}" for i in range(len(d))])
    d["phone_number"] = "+234000"
    d = pd.concat([d, d.iloc[[12]]])  # identifiant en double, comme dans le jeu réel
    d.to_csv(tmp_path / "clients.csv", index=False)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("CLIENTS_PATH", str(tmp_path / "clients.csv"))
    from src.api import app
    with TestClient(app) as c:
        assert c.get("/ui").status_code == 200 and "risque" in c.get("/ui").text
        r = c.get("/clients/cust12")
        assert r.status_code == 200
        body = r.json()
        assert body["customer_id"] == "CUST12" and 0 <= body["default_risk_score"] <= 1
        assert body["risk"] in ("élevé", "faible") and "phone_number" not in body["profile"]
        assert c.get("/clients/INCONNU").status_code == 404
        assert "CUST12" in c.get("/clients", params={"q": "CUST12"}).json()
