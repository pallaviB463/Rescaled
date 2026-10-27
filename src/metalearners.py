"""metalearners.py  (OWNER: Member Q).
Every learner has the SAME signature  learner(M, X, W, y, Xte, e=0.5) -> tau_hat (array over Xte).
M is a model class with fit(X, y) -> self and predict(X). e = known treatment probability
(0.5 in simulation; computed from the data on Hillstrom; Criteo is heavily treated, pass its share)."""
import numpy as np
from sklearn.model_selection import KFold
from models import L2


def t_learner(M, X, W, y, Xte, e=0.5):  # e unused, kept for a uniform interface
    return M().fit(X[W == 1], y[W == 1]).predict(Xte) - M().fit(X[W == 0], y[W == 0]).predict(Xte)


def s_learner(M, X, W, y, Xte, e=0.5):  # e unused, kept for a uniform interface
    m = M().fit(np.column_stack([X, W]), y)
    n = len(Xte)
    return m.predict(np.column_stack([Xte, np.ones(n)])) - m.predict(np.column_stack([Xte, np.zeros(n)]))


def x_learner(M, X, W, y, Xte, e=0.5):
    m1, m0 = M().fit(X[W == 1], y[W == 1]), M().fit(X[W == 0], y[W == 0])
    d1 = y[W == 1] - m0.predict(X[W == 1])   # imputed effect for treated
    d0 = m1.predict(X[W == 0]) - y[W == 0]   # imputed effect for controls
    # second stage is plain L2: imputed effects can be negative, so count models do not apply here
    t1 = L2().fit(X[W == 1], d1).predict(Xte)
    t0 = L2().fit(X[W == 0], d0).predict(Xte)
    return (1 - e) * t1 + e * t0             # weight by propensity (Kunzel et al.)


def dr_learner(M, X, W, y, Xte, e=0.5, k=5, seed=0):
    """Cross-fitted doubly robust pseudo-outcome, then L2 regression on X. A row is never
    predicted by a nuisance model that was trained on it."""
    phi = np.zeros(len(y))
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        w = W[tr]
        m1 = M().fit(X[tr][w == 1], y[tr][w == 1])
        m0 = M().fit(X[tr][w == 0], y[tr][w == 0])
        mu1, mu0 = m1.predict(X[te]), m0.predict(X[te])
        phi[te] = mu1 - mu0 + W[te] * (y[te] - mu1) / e - (1 - W[te]) * (y[te] - mu0) / (1 - e)
    return L2().fit(X, phi).predict(Xte)


def to_learner(M, X, W, y, Xte, e=0.5):  # transformed-outcome baseline (COULD tier); M unused
    return L2().fit(X, y * (W - e) / (e * (1 - e))).predict(Xte)


LEARNERS = {"S": s_learner, "T": t_learner, "X": x_learner, "DR": dr_learner}
