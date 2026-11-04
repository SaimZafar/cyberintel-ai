import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from xgboost import XGBClassifier

D = "data/processed/"
SEEDS = [0, 1, 2, 3, 4]
K = 10
cfg = json.load(open("models/classifier_v1.json"))["config"]


def load(name):
    X = pd.read_parquet(D + f"X_{name}.parquet")
    y = pd.read_parquet(D + f"y_{name}.parquet")["y"].to_numpy()
    g = pd.read_parquet(D + f"{name}.parquet")["group"].astype("uint64").to_numpy()
    return X, y, g


Xf, yf, gf = load("fit")
Xv, yv, gv = load("val")
Xt, yt, gt = load("test_eval")

X_tr = pd.concat([Xf, Xv], ignore_index=True)       # the official training partition
y_tr, g_tr = np.r_[yf, yv], np.r_[gf, gv]
X_all = pd.concat([X_tr, Xt], ignore_index=True)    # both partitions pooled
y_all, g_all = np.r_[y_tr, yt], np.r_[g_tr, gt]
n_tr = len(y_tr)
print("training partition:", n_tr, "| test partition:", len(yt), "| pooled:", len(y_all))


def random_split(y, seed):
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)


def group_split(y, g, seed):
    codes = pd.factorize(g)[0]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(sgkf.split(np.zeros(len(y)), y, groups=codes))


def evaluate(X, y, g, tr, te, seed):
    counts = np.bincount(y[tr], minlength=K)
    sw = ((counts.max() / np.maximum(counts, 1)) ** 0.5)[y[tr]]
    model = XGBClassifier(tree_method="hist", n_jobs=6, random_state=seed, **cfg)
    model.fit(X.iloc[tr], y[tr], sample_weight=sw)
    pred = model.predict(X.iloc[te])
    yy, att = y[te], y[te] != 0
    support = np.bincount(yy, minlength=K)
    f1 = f1_score(yy, pred, labels=list(range(K)), average=None, zero_division=0)
    return {
        "n_test": len(te),
        "seen_share": float(np.isin(g[te], g[tr]).mean()),
        "accuracy": float((pred == yy).mean()),
        "macro_f1": float(f1[support > 0].mean()),
        "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
        "normal_recall": float((pred[~att] == 0).mean()),
        "f1_DoS": f1[4], "f1_Analysis": f1[6], "f1_Backdoor": f1[7],
    }


rows = []
for seed in SEEDS:
    jobs = {
        "1 pooled, random": (X_all, y_all, g_all, *random_split(y_all, seed)),
        "2 pooled, group-aware": (X_all, y_all, g_all, *group_split(y_all, g_all, seed)),
        "3 train partition, random": (X_tr, y_tr, g_tr, *random_split(y_tr, seed)),
        "4 train partition, group-aware": (X_tr, y_tr, g_tr, *group_split(y_tr, g_tr, seed)),
        "5 official train->test": (X_all, y_all, g_all, np.arange(n_tr), np.arange(n_tr, len(y_all))),
    }
    for name, (X, y, g, tr, te) in jobs.items():
        r = evaluate(X, y, g, tr, te, seed)
        r.update({"protocol": name, "seed": seed})
        rows.append(r)
        print(f"seed {seed}  {name:<32} macro_f1 {r['macro_f1']:.3f}  seen {r['seen_share']:.2f}", flush=True)

raw = pd.DataFrame(rows)
raw.to_csv("reports/protocol_compare_raw.csv", index=False)

metrics = ["seen_share", "accuracy", "macro_f1", "binary_f1", "normal_recall",
           "f1_DoS", "f1_Analysis", "f1_Backdoor"]
mean, std = raw.groupby("protocol")[metrics].mean(), raw.groupby("protocol")[metrics].std()
summary = mean.round(3).astype(str) + " +/- " + std.round(3).astype(str)
summary.insert(0, "n_test", raw.groupby("protocol")["n_test"].first())
summary.to_csv("reports/protocol_compare_summary.csv")
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)
print()
print(summary.to_string())
print("\nProtocol 5: the split is fixed, so its spread is model randomness only.")