import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from engine import Engine
from features import CLASSES

OUT = Path("reports/final_test_results.csv")
if OUT.exists() and "--again" not in sys.argv:
    sys.exit("Final test results already exist. Do not tune against them. "
             "Use --again only if the data or the model really changed.")

D = "data/processed/"
X = pd.read_parquet(D + "X_test_eval.parquet")
y = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
meta = pd.read_parquet(D + "test_eval.parquet")[["group", "seen_in_train"]]
K = len(CLASSES)

eng = Engine()
proba = eng.clf.predict_proba(X)
pred = proba.argmax(axis=1)
dec = eng.decide(X)
alert = dec["severity_level"].isin(["High", "Critical"]).to_numpy()
incident = dec["incident"].to_numpy()

seen = meta["seen_in_train"].to_numpy()
first = ~meta["group"].duplicated().to_numpy()
subsets = {
    "all test rows": np.arange(len(y)),
    "unseen in training": np.where(~seen)[0],
    "unseen and distinct": np.where(~seen & first)[0],
}


def confusion(yy, pp):
    return np.bincount(yy * K + pp, minlength=K * K).reshape(K, K)


def per_class(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1, cm.sum(1)


def metrics(idx):
    yy, pp = y[idx], pred[idx]
    cm = confusion(yy, pp)
    prec, rec, f1, support = per_class(cm)
    present = support > 0
    att, pa = yy != 0, pp != 0
    tp = (att & pa).sum()
    a = alert[idx]
    return {
        "accuracy": np.trace(cm) / cm.sum(),
        "macro_f1": f1[present].mean(),
        "weighted_f1": (f1 * support).sum() / support.sum(),
        "binary_attack_f1": 2 * tp / max(2 * tp + (~att & pa).sum() + (att & ~pa).sum(), 1),
        "alert_recall": a[att].mean(),
        "alert_precision": att[a].mean() if a.any() else 0.0,
        "false_alerts_per_1000_normal": 1000 * a[~att].mean(),
        "normal_becomes_incident_pct": 100 * incident[idx][~att].mean(),
    }


def aucs(idx):
    yy, P = y[idx], proba[idx]
    roc, ap = [], []
    for k in range(K):
        pos = yy == k
        if pos.sum() == 0 or pos.all():
            continue
        roc.append(roc_auc_score(pos, P[:, k]))
        ap.append(average_precision_score(pos, P[:, k]))
    return float(np.mean(roc)), float(np.mean(ap))


def resample(idx, rng):
    """Stratified bootstrap: resample inside each true class."""
    parts = []
    for k in range(K):
        ik = idx[y[idx] == k]
        if len(ik):
            parts.append(rng.choice(ik, len(ik), replace=True))
    return np.concatenate(parts)


B, B_AUC = 1000, 200      # AUC is slower, so it uses fewer resamples
rng = np.random.default_rng(42)
rows, classes = [], []
for name, idx in subsets.items():
    print(f"running: {name} ({len(idx)} rows)", flush=True)
    point = metrics(idx)
    point["macro_roc_auc"], point["macro_pr_auc"] = aucs(idx)

    boots = []
    for b in range(B):
        bi = resample(idx, rng)
        m = metrics(bi)
        if b < B_AUC:
            m["macro_roc_auc"], m["macro_pr_auc"] = aucs(bi)
        boots.append(m)
    boots = pd.DataFrame(boots)

    for key, value in point.items():
        lo, hi = np.percentile(boots[key].dropna(), [2.5, 97.5])
        rows.append({"subset": name, "n": len(idx), "metric": key,
                     "value": value, "ci_low": lo, "ci_high": hi})

    prec, rec, f1, support = per_class(confusion(y[idx], pred[idx]))
    for k, c in enumerate(CLASSES):
        classes.append({"subset": name, "class": c, "support": int(support[k]),
                        "precision": prec[k], "recall": rec[k], "f1": f1[k]})

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
pd.DataFrame(classes).to_csv("reports/final_test_per_class.csv", index=False)

for name in subsets:
    print(f"\n=== {name} ===")
    for _, r in res[res["subset"] == name].iterrows():
        print(f"  {r['metric']:<30}{r['value']:9.4f}   95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

pc = pd.DataFrame(classes)
for name in ["all test rows", "unseen in training"]:
    print(f"\nPER CLASS: {name}")
    print(pc[pc["subset"] == name].drop(columns="subset").round(3).to_string(index=False))