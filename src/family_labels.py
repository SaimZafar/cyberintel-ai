import numpy as np
import pandas as pd
from engine import Engine
from features import CLASSES

D = "data/processed/"
eng = Engine()
FAMILY = {c: "Exploit-type" for c in ["Exploits", "DoS", "Backdoor", "Analysis"]}
fam = np.array([FAMILY.get(c, c) for c in CLASSES])


def evaluate(X, y, thr, label):
    P = eng.clf.predict_proba(X)
    p_att = 1.0 - P[:, 0]
    alert = (p_att > thr) & (y != 0)            
    name = P[:, 1:].argmax(axis=1) + 1
    conf = P[:, 1:].max(axis=1) / np.maximum(p_att, 1e-9)  
    print(f"\n{label}: {int(alert.sum())} alerted attacks")
    print(f"{'tau':>5}{'named':>8}{'name right':>12}{'family right (rest)':>20}{'useful overall':>16}")
    for tau in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        c, u = alert & (conf >= tau), alert & (conf < tau)
        named = c.sum() / alert.sum()
        name_ok = (name[c] == y[c]).mean() if c.any() else np.nan
        fam_ok = (fam[name[u]] == fam[y[u]]).mean() if u.any() else np.nan
        useful = ((name[c] == y[c]).sum() + (fam[name[u]] == fam[y[u]]).sum()) / alert.sum()
        print(f"{tau:5.1f}{named:8.3f}{name_ok:12.3f}{fam_ok:20.3f}{useful:16.3f}")


Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
t_val = float(np.quantile(1.0 - eng.clf.predict_proba(Xv[yv == 0])[:, 0], 1 - 20 / 1000))
evaluate(Xv, yv, t_val, "VALIDATION (threshold for 20 false alerts per 1000)")

Xd = pd.read_parquet(D + "X_demo_pool.parquet")
yd = pd.read_parquet(D + "y_demo_pool.parquet")["y"].to_numpy()
evaluate(Xd, yd, eng.alert_threshold, "DEMO POOL (unseen events, service threshold)")