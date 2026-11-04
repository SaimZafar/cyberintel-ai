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
a_val = eng.anomaly(Xv)                          # anomaly score: never saw any attack
p_full = 1.0 - eng.clf.predict_proba(Xv)[:, 0]   # full classifier: saw every class


def thr(s, budget):
    """Threshold giving `budget` false alerts per 1000 Normal validation events."""
    return float(np.quantile(s[yv == 0], 1 - budget / 1000.0))


rows = []
for c in range(1, len(CLASSES)):
    keep = yf != c
    left = [k for k in range(len(CLASSES)) if k != c]      # Normal stays index 0
    remap = {k: i for i, k in enumerate(left)}
    y_r = np.array([remap[k] for k in yf[keep]])
    counts = np.bincount(y_r)
    sw = ((counts.max() / counts) ** 0.5)[y_r]
    m = XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cfg)
    m.fit(Xf[keep], y_r, sample_weight=sw)
    p = 1.0 - m.predict_proba(Xv)[:, 0]

    target = yv == c
    normal = yv == 0
    alert_p = p > thr(p, 20)
    alert_a = a_val > thr(a_val, 20)
    union = (p > thr(p, 10)) | (a_val > thr(a_val, 10))
    rows.append({
        "held-out class": CLASSES[c],
        "val rows": int(target.sum()),
        "class seen in training": (p_full[target] > thr(p_full, 20)).mean(),
        "classifier, class unseen": alert_p[target].mean(),
        "anomaly score only": alert_a[target].mean(),
        "both (10+10 budget)": union[target].mean(),
        "both: false alerts per 1000": 1000 * union[normal].mean(),
    })
    print("done:", CLASSES[c], flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/leave_one_out.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print()
print(res.round(3).to_string(index=False))