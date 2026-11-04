import joblib
import numpy as np
import pandas as pd

D = "data/processed/"
BUDGET = 20
ALPHA = BUDGET / 1000
DRAWS = 300
rng = np.random.default_rng(0)

clf = joblib.load("models/classifier_v1.joblib")
Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
Xt = pd.read_parquet(D + "X_test_eval.parquet")
yt = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
gt = pd.read_parquet(D + "test_eval.parquet")["group"].astype("uint64").to_numpy()
pv, pt = clf.predict_proba(Xv), clf.predict_proba(Xt)


def ece_top_label(P, y, bins=15):
    """Top-label expected calibration error, equal-mass bins (as in Alalawi 2026)."""
    conf, hit = P.max(1), (P.argmax(1) == y).astype(float)
    order = np.argsort(conf)
    e = 0.0
    for idx in np.array_split(order, bins):
        e += len(idx) / len(y) * abs(hit[idx].mean() - conf[idx].mean())
    return e


print("1. TOP-LABEL ECE of classifier v1 (15 equal-mass bins)")
print(f"   validation fold (same partition): {ece_top_label(pv, yv):.3f}")
print(f"   official test partition:          {ece_top_label(pt, yt):.3f}")
print("   Alalawi 2026, UNSW-NB15, untuned XGBoost: in-distribution 0.011, shifted 0.043")

# ---- 2. how many labeled target Normal rows restore the false-alert budget ----
att_t = yt != 0
a_t = 1 - pt[:, 0]
dev, fin = (gt % 2 == 0), (gt % 2 == 1)
pool = a_t[dev & ~att_t]                       # labeled Normal rows we may draw from
fin_norm, fin_att = a_t[fin & ~att_t], a_t[fin & att_t]


def plain(s):
    return float(np.quantile(s, 1 - ALPHA))


def conformal(s):
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    return np.inf if k > n else float(np.sort(s)[k - 1])


a_v = 1 - pv[:, 0]
t_val = plain(a_v[yv == 0])
print(f"\n2. FALSE ALERTS PER 1000 NORMAL on the held-out odd half (budget {BUDGET}); pool of {len(pool)} labeled Normal rows")
print(f"   threshold from {int((yv == 0).sum())} validation Normal rows (other partition): "
      f"{1000 * (fin_norm > t_val).mean():.0f} false alerts, recall {(fin_att > t_val).mean():.3f}")
rows = []
for n in [20, 50, 100, 200, 500, 1000, 2000, 5000]:
    if n > len(pool):
        continue
    for name, fn in [("plain quantile", plain), ("conformal", conformal)]:
        fa, rc = [], []
        for _ in range(DRAWS):
            t = fn(rng.choice(pool, n, replace=False))
            fa.append(1000 * (fin_norm > t).mean())
            rc.append((fin_att > t).mean())
        fa = np.array(fa)
        rows.append({"labeled Normals": n, "method": name, "mean FA": fa.mean(), "median FA": np.median(fa),
                     "5th-95th pct FA": f"{np.percentile(fa, 5):.0f}-{np.percentile(fa, 95):.0f}",
                     "share within 10-30": float(((fa >= 10) & (fa <= 30)).mean()),
                     "median recall": np.median(rc)})
res = pd.DataFrame(rows)
res.to_csv("reports/threshold_curve.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
print("\nDraws come from one finite pool, so spread at 2000 and 5000 is understated.")