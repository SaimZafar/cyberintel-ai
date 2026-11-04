import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from engine import Engine
from features import CLASSES

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
cfg = json.load(open("models/classifier_v1.json"))["config"]

eng = Engine()
a = eng.anomaly(Xv)
normal = yv == 0


def thr(s, budget):
    return float(np.quantile(s[normal], 1 - budget / 1000.0))


rows = []
for c in range(1, len(CLASSES)):
    keep = yf != c
    left = [k for k in range(len(CLASSES)) if k != c]
    remap = {k: i for i, k in enumerate(left)}
    y_r = np.array([remap[k] for k in yf[keep]])
    counts = np.bincount(y_r)
    sw = ((counts.max() / counts) ** 0.5)[y_r]
    m = XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cfg)
    m.fit(Xf[keep], y_r, sample_weight=sw)
    p = 1.0 - m.predict_proba(Xv)[:, 0]

    t = yv == c
    clf20 = p > thr(p, 20)
    clf25 = p > thr(p, 25)
    mix = clf20 | (a > thr(a, 5))
    rows.append({
        "held-out class": CLASSES[c],
        "classifier at 20": clf20[t].mean(),
        "classifier at 25": clf25[t].mean(),
        "classifier 20 + anomaly 5": mix[t].mean(),
        "false alerts (clf 25)": 1000 * clf25[normal].mean(),
        "false alerts (mix)": 1000 * mix[normal].mean(),
    })
    print("done:", CLASSES[c], flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/leave_one_out2.csv", index=False)
pd.set_option("display.width", 250)
print()
print(res.round(3).to_string(index=False))