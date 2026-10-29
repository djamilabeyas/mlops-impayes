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
