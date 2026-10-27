import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from metalearners import LEARNERS
from metrics import rank_corr


class SK:   # model class independent of LightGBM, so these tests run anywhere
    def fit(self, X, y):
        self.m = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1, early_stopping=False, random_state=0).fit(X, y); return self
    def predict(self, X): return self.m.predict(X)


def toy(n=6000, effect=True, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)); W = rng.binomial(1, 0.5, n)
    tau = X[:, 0] if effect else np.zeros(n)
    y = X[:, 1] + W * tau + 0.1 * rng.normal(size=n)
    return X, W, y, tau


def test_uniform_signature_and_shape():
    X, W, y, _ = toy(1500)
    for name, L in LEARNERS.items():
        out = L(SK, X, W, y, X[:200], e=0.5)
        assert out.shape == (200,), name


def test_learners_recover_effect_ranking():
    X, W, y, tau = toy()
    for name in ("T", "DR", "X", "S"):
        # S-learner with trees is a known weak case (may shrink effect to zero), so it gets a looser bar
        bar = 0.5 if name == "S" else 0.9
        rho = rank_corr(LEARNERS[name](SK, X, W, y, X, e=0.5), tau)
        assert rho > bar, (name, rho)


def test_null_effect_mean_near_zero():
    X, W, y, _ = toy(effect=False)
    for name in ("T", "DR", "X", "S"):
        assert abs(np.mean(LEARNERS[name](SK, X, W, y, X, e=0.5))) < 0.1, name


def test_dr_crossfitting_never_predicts_a_row_it_trained_on():
    X, W, y, _ = toy(2000)
    class Rec(SK):
        def fit(self, X, y): self.seen = {r.tobytes() for r in X}; return super().fit(X, y)
        def predict(self, X):
            assert not any(r.tobytes() in self.seen for r in X), "leak: predicted a row used in training"
            return super().predict(X)
    # L2 (second stage) is imported inside metalearners and is not Rec, so only nuisance models are checked
    LEARNERS["DR"](Rec, X, W, y, X[:50], e=0.5)


def test_x_learner_uses_e_weights():
    X, W, y, _ = toy(2000)
    a = LEARNERS["X"](SK, X, W, y, X[:100], e=0.2)
    b = LEARNERS["X"](SK, X, W, y, X[:100], e=0.8)
    assert not np.allclose(a, b)
