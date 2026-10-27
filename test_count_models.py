
import numpy as np

from src.dgp import simulate
from src.metrics import rank_corr, topk_overlap, policy_value, regret
from src.models import Tweedie, Hurdle


# Training data
X, W, Y, _ = simulate(
    n=2000,
    scenario="opposed",
    zero_rate=0.9,
    seed=42,
)

# Independent evaluation population with exact ground truth
X_test, _, _, truth_test = simulate(
    n=2000,
    scenario="opposed",
    zero_rate=0.9,
    seed=43,
)

tau_true = truth_test["tau_level"]

X0, Y0 = X[W == 0], Y[W == 0]
X1, Y1 = X[W == 1], Y[W == 1]

for name, Model in [
    ("Tweedie", Tweedie),
    ("Hurdle", Hurdle),
]:
    print(f"\n{'=' * 40}")
    print(f"Model: {name}")
    print("=" * 40)

    model0 = Model().fit(X0, Y0)
    model1 = Model().fit(X1, Y1)

    tau_hat = model1.predict(X_test) - model0.predict(X_test)

    print("Predictions finite:", np.isfinite(tau_hat).all())
    print("Mean estimated CATE:", round(float(tau_hat.mean()), 4))
    print("Rank correlation:", round(rank_corr(tau_hat, tau_true), 4))
    print("Top-20% overlap:", round(topk_overlap(tau_hat, tau_true), 4))
    print("Policy value:", round(policy_value(tau_hat, tau_true), 4))
    print("Regret:", round(regret(tau_hat, tau_true), 4))
