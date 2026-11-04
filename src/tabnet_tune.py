import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
Xf = pd.read_parquet(D + "X_fit.parquet").to_numpy(dtype=np.float32)      # folds 1 to 4
Xv = pd.read_parquet(D + "X_val.parquet").to_numpy(dtype=np.float32)      # fold 0
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
counts = np.bincount(yf, minlength=K)
sample_w = {i: float((counts.max() / counts[i]) ** 0.5) for i in range(K)}
att = yv != 0


class MacroF1(Metric):
    def __init__(self):
        self._name = "macro_f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return f1_score(y_true, np.argmax(y_score, axis=1), average="macro")


old = [dict(n_d=16, n_a=16, n_steps=3, gamma=1.3, lr=2e-2, batch=1024, lambda_sparse=1e-4),
       dict(n_d=32, n_a=32, n_steps=4, gamma=1.5, lr=1e-2, batch=1024, lambda_sparse=1e-4),
       dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, lr=2e-2, batch=1024, lambda_sparse=1e-4)]
rows = []
prev = Path("reports/tabnet_results.csv")
if prev.exists():
    p = pd.read_csv(prev)
    p = p[p["seed"] == 0]
    for i, cfg in enumerate(old):
        hit = p[p["config"] == i]
        if len(hit):
            rows.append({"id": i, **cfg, "macro_f1": float(hit["macro_f1"].iloc[0]),
                         "binary_f1": float(hit["binary_f1"].iloc[0]),
                         "normal_recall": float(hit["normal_recall"].iloc[0])})
else:
    print("reports/tabnet_results.csv not found: the three earlier configs will not be counted")

rng = np.random.RandomState(7)
new = []
for _ in range(11):
    nd = int(rng.choice([8, 16, 24, 32]))
    new.append(dict(n_d=nd, n_a=nd, n_steps=int(rng.choice([3, 4, 5])),
                    gamma=float(rng.choice([1.0, 1.3, 1.5, 2.0])),
                    lr=float(rng.choice([5e-3, 1e-2, 2e-2, 3e-2])),
                    batch=int(rng.choice([1024, 2048])),
                    lambda_sparse=float(rng.choice([0.0, 1e-4, 1e-3]))))


def fit(cfg, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    clf = TabNetClassifier(n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"], gamma=cfg["gamma"],
                           lambda_sparse=cfg["lambda_sparse"], seed=seed, device_name="cpu", verbose=0,
                           optimizer_params=dict(lr=cfg["lr"]))
    clf.fit(Xf, yf, eval_set=[(Xv, yv)], eval_name=["val"], eval_metric=[MacroF1],
            max_epochs=25, patience=5, batch_size=cfg["batch"], virtual_batch_size=128,
            weights=sample_w, drop_last=False)
    return clf


for j, cfg in enumerate(new):
    i = j
    t0 = time.time()
    clf = fit(cfg)
    pred = clf.predict(Xv)
    f1 = f1_score(yv, pred, average="macro")
    rows.append({"id": i, **cfg, "macro_f1": f1,
                 "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
                 "normal_recall": float((pred[~att] == 0).mean()), "seconds": time.time() - t0})
    print(f"tabnet config {i}: val macro F1 {f1:.4f} ({(time.time() - t0) / 60:.1f} min) {cfg}", flush=True)
    del clf

res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
res.to_csv("reports/tabnet_tuning.csv", index=False)
best = res.iloc[0]
json.dump({"config": {k: (int(best[k]) if k in ("n_d", "n_a", "n_steps", "batch") else float(best[k]))
                      for k in ("n_d", "n_a", "n_steps", "gamma", "lr", "batch", "lambda_sparse")},
           "val_macro_f1": float(best["macro_f1"])}, open("models/tabnet_best.json", "w"), indent=2)
pd.set_option("display.width", 250)
print(f"\nconfigs counted: {len(res)}")
print(res.round(4).head(6).to_string(index=False))
print("\nXGBoost 0.6567 | LightGBM 0.6551 | random forest 0.6481 | MLP 0.5780 (fold 0, best of 11 each)")