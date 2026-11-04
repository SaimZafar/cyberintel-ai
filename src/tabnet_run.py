import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
Xf = pd.read_parquet(D + "X_fit.parquet").to_numpy(dtype=np.float32)
Xv = pd.read_parquet(D + "X_val.parquet").to_numpy(dtype=np.float32)
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
counts = np.bincount(yf, minlength=K)
sample_w = {i: float((counts.max() / counts[i]) ** 0.5) for i in range(K)}   # square-root class sampling


class MacroF1(Metric):
    def __init__(self):
        self._name = "macro_f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return f1_score(y_true, np.argmax(y_score, axis=1), average="macro")


configs = [
    dict(n_d=16, n_a=16, n_steps=3, gamma=1.3, lr=2e-2, batch=1024),
    dict(n_d=32, n_a=32, n_steps=4, gamma=1.5, lr=1e-2, batch=1024),
    dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, lr=2e-2, batch=1024),
]


def fit(cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    clf = TabNetClassifier(n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"], gamma=cfg["gamma"],
                           lambda_sparse=1e-4, seed=seed, device_name="cpu", verbose=1,
                           optimizer_params=dict(lr=cfg["lr"]))
    t0 = time.time()
    clf.fit(Xf, yf, eval_set=[(Xv, yv)], eval_name=["val"], eval_metric=[MacroF1],
            max_epochs=25, patience=5, batch_size=cfg["batch"], virtual_batch_size=128,
            weights=sample_w, drop_last=False)
    return clf, time.time() - t0


def metrics(pred):
    att = yv != 0
    return {"macro_f1": f1_score(yv, pred, average="macro"),
            "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
            "normal_recall": float((pred[~att] == 0).mean())}


rows, best = [], (-1, None, None)
for i, cfg in enumerate(configs):
    print(f"\n=== config {i}: {cfg} ===", flush=True)
    clf, sec = fit(cfg, seed=0)
    m = metrics(clf.predict(Xv))
    rows.append({"config": i, "seed": 0, **cfg, **m, "seconds": sec})
    print(f"config {i}: val macro F1 {m['macro_f1']:.4f} ({sec / 60:.1f} min)", flush=True)
    if m["macro_f1"] > best[0]:
        best = (m["macro_f1"], i, clf)

best_i = best[1]
print(f"\nbest config: {best_i}; rerunning with seeds 1 and 2", flush=True)
final = [rows[best_i]]
for s in (1, 2):
    clf, sec = fit(configs[best_i], seed=s)
    m = metrics(clf.predict(Xv))
    final.append({"config": best_i, "seed": s, **configs[best_i], **m, "seconds": sec})
    print(f"  seed {s}: {m['macro_f1']:.4f}", flush=True)

pd.DataFrame(rows + final[1:]).to_csv("reports/tabnet_results.csv", index=False)
fin = pd.DataFrame(final)
print(f"\nbest TabNet config, 3 seeds: macro F1 {fin['macro_f1'].mean():.4f} +/- {fin['macro_f1'].std():.4f} "
      f"| binary F1 {fin['binary_f1'].mean():.4f} | Normal recall {fin['normal_recall'].mean():.4f}")
print("XGBoost v1 (validation):     macro F1 0.6567 | binary F1 0.9666 | Normal recall 0.8838")

path = best[2].save_model("models/tabnet_best")
print("saved:", path, f"({os.path.getsize(path) / 1e6:.2f} MB)")
torch.set_num_threads(1)
row = Xv[:1]
for _ in range(10):
    best[2].predict_proba(row)
t = []
for _ in range(100):
    a = time.perf_counter()
    best[2].predict_proba(row)
    t.append((time.perf_counter() - a) * 1000)
print(f"single-row predict_proba, 1 thread: {np.median(t):.2f} ms (median)")