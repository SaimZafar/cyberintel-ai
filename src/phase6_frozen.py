import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from features import CLASSES

OUT = Path("reports/phase6_frozen.csv")
if OUT.exists() and "--again" not in sys.argv:
    sys.exit("Already run once. Do not tune against this. Use --again only if data or model changed.")

D = "data/processed/"
K = len(CLASSES)
X = pd.read_parquet(D + "X_test_eval.parquet")
y = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
g = pd.read_parquet(D + "test_eval.parquet")["group"].astype("uint64")
odd = (g % 2 == 1).to_numpy()                     # the half not used to pick 0.956
gcode = pd.factorize(g)[0]
att = y != 0

# thresholds, fixed before this run
y_dev = np.r_[pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy(),
              pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()]
p_oof = 1.0 - np.load("reports/cv/xgb_oof_proba.npy").astype(np.float64)[:, 0]
T_OOF = float(np.quantile(p_oof[y_dev == 0], 0.98))          # out-of-fold, budget 20 per 1000
T_REC = 0.956                                                  # chosen on the other half of this partition
print(f"thresholds: out-of-fold {T_OOF:.3f} | target-recalibrated {T_REC}")

clf = joblib.load("models/classifier_v1.joblib")
clf.set_params(n_jobs=6)
proba = clf.predict_proba(X)
pred = proba.argmax(axis=1)
p = 1.0 - proba[:, 0]

rng = np.random.default_rng(0)
G = gcode.max() + 1
boots = [np.bincount(rng.integers(0, G, G), minlength=G)[gcode].astype(float) for _ in range(500)]
ones = np.ones(len(y))


def macro_f1(w):
    cm = np.bincount(y * K + pred, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    pr, rc = tp / np.maximum(cm.sum(0), 1e-9), tp / np.maximum(cm.sum(1), 1e-9)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return f1[cm.sum(1) > 0].mean()


def rate(a, mask, w):
    m = mask * w
    return (a * m).sum() / m.sum()


rows = []


def add(name, fn):
    pt, b = fn(ones), [fn(w) for w in boots]
    lo, hi = np.percentile(b, [2.5, 97.5])
    rows.append({"measure": name, "value": pt, "ci_low": lo, "ci_high": hi})
    print(f"{name:<70}{pt:9.4f}   95% CI [{lo:.4f}, {hi:.4f}]", flush=True)


add("multiclass macro F1, full test partition", macro_f1)
add("multiclass accuracy, full test partition", lambda w: rate(pred == y, np.ones(len(y), bool), w))

rules = [("out-of-fold threshold", T_OOF, "full partition", np.ones(len(y), bool)),
         ("out-of-fold threshold", T_OOF, "odd half", odd),
         ("recalibrated 0.956", T_REC, "odd half", odd)]
for name, t, mname, mask in rules:
    a = p > t
    add(f"{name} | {mname}: false alerts per 1000 Normal",
        lambda w, a=a, mask=mask: 1000 * rate(a, (~att) & mask, w))
    add(f"{name} | {mname}: attack recall", lambda w, a=a, mask=mask: rate(a, att & mask, w))

true = pd.Series([CLASSES[k] for k in y])
share = {}
for name, t, mname, mask in rules:
    a = pd.Series(p > t)
    share[f"{name} | {mname}"] = a[mask].groupby(true[mask]).mean().reindex(CLASSES)
print("\nSHARE OF EACH TRUE CLASS THAT ALERTS")
print(pd.DataFrame(share).round(3).to_string())

pd.DataFrame(rows).to_csv(OUT, index=False)
pd.DataFrame(share).to_csv("reports/phase6_per_class_alert_share.csv")