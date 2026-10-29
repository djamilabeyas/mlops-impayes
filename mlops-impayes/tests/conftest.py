import numpy as np, pandas as pd, pytest

@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 3000
    d = pd.DataFrame({"tenure_months": rng.integers(1, 120, n),
                      "avg_spend": rng.gamma(2, 50, n),
                      "recharge_freq": rng.poisson(5, n),
                      "payment_consistency": rng.uniform(0, 1, n),
                      "late_payments": rng.poisson(1.5, n),
                      "region": rng.choice(["N", "S", "E", "W"], n)})
    z = 1.2 * d.late_payments - 3 * d.payment_consistency - 0.01 * d.tenure_months
    d["default_risk"] = (z + rng.normal(0, 0.5, n) > np.quantile(z, 0.8)).astype(int)
    return d

@pytest.fixture
def params():
    from src.config import load_params
    p = load_params()
    p["gate"] = {"min_f1": 0.0, "min_auc": 0.0}
    return p
