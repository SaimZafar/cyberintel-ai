import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
MODELS = ["xgb", "lgbm", "rf", "mlp", "tabnet"]
B = 1000

fit_raw = pd.read_parquet(D + "fit.parquet")
val_raw = pd.read_parquet(D + "val.parquet")
y = np.r_[pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy(),
          pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()]
ids = np.r_[fit_raw["id"].to_numpy(), val_raw["id"].to_numpy()]
groups = np.r_[fit_raw["group"].to_numpy(), val_raw["group"].to_numpy()]
flags = pd.read_csv("reports/release/unsw_nb15_row_flags.csv", dtype={"group": str},
                    usecols=["partition", "id", "train_fold_group_aware"])
flags = flags[flags["partition"] == "train"].set_index("id")["train_fold_group_aware"]
fold = flags.loc[ids].to_numpy()

m = fold >= 1                                    # folds 1 to 4: not used for any selection
ym = y[m]
gcode = pd.factorize(groups[m])[0]
G = gcode.max() + 1
preds = {k: np.load(f"reports/cv/{k}_oof.npy")[m].astype(int) for k in MODELS}
probs = {k: np.load(f"reports/cv/{k}_oof_proba.npy")[m].astype(np.float64) for k in MODELS}
assert all((p >= 0).all() for p in preds.values()), "missing out-of-fold predictions"
print("rows:", len(ym), "| identical-row groups:", G)


def macro_f1(pred, w):
    cm = np.bincount(ym * K + pred, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(0), 1e-9)
    rec = tp / np.maximum(cm.sum(1), 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return f1[cm.sum(1) > 0].mean()


# ---- 1. macro F1 with cluster bootstrap, and difference from XGBoost on the same resamples ----
rng = np.random.default_rng(0)
ones = np.ones(len(ym))
point = {k: macro_f1(preds[k], ones) for k in MODELS}
boot = {k: [] for k in MODELS}
for _ in range(B):
    w = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)[gcode]
    for k in MODELS:
        boot[k].append(macro_f1(preds[k], w))
rows = []
for k in MODELS:
    b = np.array(boot[k])
    d = b - np.array(boot["xgb"])
    lo, hi = np.percentile(b, [2.5, 97.5])
    dlo, dhi = np.percentile(d, [2.5, 97.5])
    rows.append({"model": k, "macro_F1": point[k], "CI_low": lo, "CI_high": hi,
                 "diff_vs_xgb": point[k] - point["xgb"], "diff_CI_low": dlo, "diff_CI_high": dhi,
                 "share_better_than_xgb": float((d > 0).mean())})
t1 = pd.DataFrame(rows)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n1. MACRO F1, cluster bootstrap over identical-row groups (%d resamples)" % B)
print(t1.round(4).to_string(index=False))

# ---- 2. threshold-free ranking quality ----
att = ym != 0
rows = []
for k in MODELS:
    p = 1.0 - probs[k][:, 0]
    roc, ap = [], []
    for c in range(K):
        pos = ym == c
        roc.append(roc_auc_score(pos, probs[k][:, c]))
        ap.append(average_precision_score(pos, probs[k][:, c]))
    rows.append({"model": k, "binary_AUROC": roc_auc_score(att, p), "binary_AP": average_precision_score(att, p),
                 "macro_OvR_AUROC": np.mean(roc), "macro_OvR_PRAUC": np.mean(ap)})
print("\n2. RANKING QUALITY (pooled folds 1 to 4)")
print(pd.DataFrame(rows).round(4).to_string(index=False))

# ---- 3. recall at fixed false-alert budgets ----
hard = ["DoS", "Analysis", "Backdoor", "Fuzzers", "Exploits"]
rows = []
for k in MODELS:
    p = 1.0 - probs[k][:, 0]
    for budget in (10, 20, 50):
        t = np.quantile(p[~att], 1 - budget / 1000.0)
        a = p > t
        r = {"model": k, "budget": budget, "realized_FA_per_1000": 1000 * a[~att].mean(),
             "attack_recall": a[att].mean()}
        for c in hard:
            r[c] = a[ym == CLASSES.index(c)].mean()
        rows.append(r)
print("\n3. RECALL AT A FALSE-ALERT BUDGET (threshold from the pooled Normal rows, same rule for every model)")
print(pd.DataFrame(rows).round(3).to_string(index=False))