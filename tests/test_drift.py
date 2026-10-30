import numpy as np
from src.monitoring.drift import psi, should_retrain

def test_psi_stable():
    x = np.random.default_rng(1).normal(0, 1, 5000)
    assert psi(x, x) < 0.01

def test_psi_shift_detected():
    r = np.random.default_rng(1)
    assert psi(r.normal(0, 1, 5000), r.normal(1.5, 1, 5000)) > 0.2

def test_should_retrain(params):
    assert should_retrain(0.3, None, params)["retrain"]
    assert should_retrain(0.05, 0.45, params)["retrain"]
    assert not should_retrain(0.05, 0.60, params)["retrain"]
