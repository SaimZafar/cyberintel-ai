import json
import time
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
TARGET = 3000

Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
fit_raw = pd.read_parquet(D + "fit.parquet")
val_raw = pd.read_parquet(D + "val.parquet")
X = pd.concat([Xf, Xv], ignore_index=True).to_numpy(dtype=np.float32)
y = np.r_[pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy(),
          pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()]
ids = np.r_[fit_raw["id"].to_numpy(), val_raw["id"].to_numpy()]
groups = np.r_[fit_raw["group"].to_numpy(), val_raw["group"].to_numpy()]
flags = pd.read_csv("reports/release/unsw_nb15_row_flags.csv", dtype={"group": str},
                    usecols=["partition", "id", "train_fold_group_aware"])
fold = flags[flags["partition"] == "train"].set_index("id")["train_fold_group_aware"].loc[ids].to_numpy()
cfg = json.load(open("models/classifier_v1.json"))


def make():
    return XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cfg["config"])


def weights(yy):
    c = np.bincount(yy, minlength=K)
    return ((c.max() / c) ** cfg["weight_power"])[yy]


def macro(yt, p):
    return f1_score(yt, p, average="macro")


# ---- 1. is the baseline overfitting? train on folds 1-4, compare with fold 0 ----
tr, te = np.where(fold != 0)[0], np.where(fold == 0)[0]
m = make().fit(X[tr], y[tr], sample_weight=weights(y[tr]))
ptr, pte = m.predict(X[tr]), m.predict(X[te])
top = pd.DataFrame({"g": groups[tr], "y": y[tr]}).groupby("g")["y"].apply(lambda s: s.value_counts().iloc[0]).sum()
print("1. TRAIN vs HELD-OUT (baseline XGBoost, trained on folds 1-4, tested on fold 0)")
print(f"   training rows:  accuracy {accuracy_score(y[tr], ptr):.4f} | macro F1 {macro(y[tr], ptr):.4f} "
      f"| in-sample ceiling for these rows {top / len(tr):.4f}")
print(f"   held-out fold:  accuracy {accuracy_score(y[te], pte):.4f} | macro F1 {macro(y[te], pte):.4f}", flush=True)

# ---- 2. oversampling inside each training fold ----
base = np.load("reports/cv/xgb_oof.npy").astype(int)


def resample(Xt, yt, kind):
    cnt = np.bincount(yt, minlength=K)
    strat = {c: TARGET for c in range(K) if cnt[c] < TARGET}
    if not strat:
        return Xt, yt
    s = (RandomOverSampler(sampling_strategy=strat, random_state=0) if kind == "ros"
         else SMOTE(sampling_strategy=strat, k_neighbors=5, random_state=0))
    return s.fit_resample(Xt, yt)


oof = {}
for kind in ["ros", "smote"]:
    pred = np.full(len(y), -1, dtype=int)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        t0 = time.time()
        Xr, yr = resample(X[tr], y[tr], kind)
        pred[te] = make().fit(Xr, yr, sample_weight=weights(yr)).predict(X[te])
        print(f"   {kind} fold {k}: macro F1 {macro(y[te], pred[te]):.4f} ({time.time() - t0:.0f} s)", flush=True)
    oof[kind] = pred

# ---- pooled folds 1-4, with a cluster bootstrap over identical-row groups ----
mk = fold >= 1
ym, gcode = y[mk], pd.factorize(groups[mk])[0]
G = gcode.max() + 1
rng = np.random.default_rng(0)
boots = [np.bincount(rng.integers(0, G, G), minlength=G)[gcode].astype(float) for _ in range(500)]


def mf1(p, w):
    cm = np.bincount(ym * K + p, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    pr, rc = tp / np.maximum(cm.sum(0), 1e-9), tp / np.maximum(cm.sum(1), 1e-9)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return f1[cm.sum(1) > 0].mean()


variants = {"baseline": base[mk], "random oversampling": oof["ros"][mk], "SMOTE": oof["smote"][mk]}
b0 = np.array([mf1(variants["baseline"], w) for w in boots])
rows = []
for name, p in variants.items():
    b = np.array([mf1(p, w) for w in boots])
    d = b - b0
    rec = {c: float((p[ym == CLASSES.index(c)] == CLASSES.index(c)).mean())
           for c in ["DoS", "Analysis", "Backdoor", "Worms", "Shellcode"]}
    rows.append({"variant": name, "macro_F1": mf1(p, np.ones(len(ym))),
                 "diff_vs_baseline": float(mf1(p, np.ones(len(ym))) - mf1(variants["baseline"], np.ones(len(ym)))),
                 "diff_CI_low": np.percentile(d, 2.5), "diff_CI_high": np.percentile(d, 97.5),
                 "accuracy": float((p == ym).mean()),
                 "Normal_recall": float((p[ym == 0] == 0).mean()), **rec})
res = pd.DataFrame(rows)
res.to_csv("reports/augment_cv.csv", index=False)
pd.set_option("display.width", 250)
print("\n2. OVERSAMPLING INSIDE TRAINING FOLDS (pooled folds 1-4)")
print(res.round(4).to_string(index=False))