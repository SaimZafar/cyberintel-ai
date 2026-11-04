import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.model_selection import GroupKFold

D = "data/processed/"
model = joblib.load("models/classifier_v1.joblib")
Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
groups = pd.read_parquet(D + "val.parquet")["group"].to_numpy()

proba = model.predict_proba(Xv)
logp = np.log(np.clip(proba, 1e-9, 1))


def apply_T(logp, T):
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(p, y):
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))


def fit_T(logp, y):
    best = minimize_scalar(lambda T: nll(apply_T(logp, T), y), bounds=(0.2, 5), method="bounded")
    return float(best.x)


def ece(p, y, bins=10):
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    correct = (pred == y)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total


def brier(p, y):
    onehot = np.eye(p.shape[1])[y]
    return np.mean(((p - onehot) ** 2).sum(axis=1))


# honest check: fit T on one half of validation, measure on the other half
cal = np.zeros_like(proba)
for a, b in GroupKFold(n_splits=2).split(Xv, yv, groups):
    T = fit_T(logp[a], yv[a])
    cal[b] = apply_T(logp[b], T)

print("same predictions after calibration:", bool((cal.argmax(1) == proba.argmax(1)).all()))
print(f"{'':>10} {'ECE':>8} {'log loss':>9} {'Brier':>8}")
print(f"{'before':>10} {ece(proba, yv):8.4f} {nll(proba, yv):9.4f} {brier(proba, yv):8.4f}")
print(f"{'after':>10} {ece(cal, yv):8.4f} {nll(cal, yv):9.4f} {brier(cal, yv):8.4f}")

# final temperature uses all of validation
T_final = fit_T(logp, yv)
print("final temperature T =", round(T_final, 3), "(T > 1 softens, T < 1 sharpens)")
with open("models/calibration.json", "w") as f:
    json.dump({"temperature": T_final}, f)

# reliability diagram: when the model says 0.9, is it right 90% of the time?
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for ax, p, title in [(axes[0], proba, "Before"), (axes[1], cal, "After")]:
    conf, correct = p.max(axis=1), (p.argmax(axis=1) == yv)
    edges = np.linspace(0, 1, 11)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 20:
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_title(f"{title} calibration")
    ax.set_xlabel("stated confidence")
    ax.legend()
axes[0].set_ylabel("actual accuracy")
plt.tight_layout()
plt.savefig("reports/figures/calibration.png", dpi=150)
print("saved reports/figures/calibration.png")