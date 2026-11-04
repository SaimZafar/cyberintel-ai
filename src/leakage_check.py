import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

D = "data/processed/"
SEEDS = [0, 1, 2]
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
X_tr = pd.concat([Xf, Xv], ignore_index=True)
y_tr, g_tr = np.r_[yf, yv], np.r_[gf, gv]
X_all = pd.concat([X_tr, Xt], ignore_index=True)
y_all, g_all = np.r_[y_tr, yt], np.r_[g_tr, gt]

rows = []
for seed in SEEDS:
    for pname, (X, y, g) in {"pooled, random": (X_all, y_all, g_all),
                             "train partition, random": (X_tr, y_tr, g_tr)}.items():
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        seen = np.isin(g[te], g[tr])
        yy = y[te]

        # lookup: the most common training label for each pattern
        cnt = (pd.DataFrame({"g": g[tr], "y": y[tr]}).groupby(["g", "y"]).size()
               .reset_index(name="n").sort_values("n", ascending=False)
               .drop_duplicates("g").set_index("g")["y"])
        lookup_acc = float((cnt.loc[g[te][seen]].to_numpy() == yy[seen]).mean())

        counts = np.bincount(y[tr], minlength=K)
        sw = ((counts.max() / np.maximum(counts, 1)) ** 0.5)[y[tr]]
        models = {
            "xgboost": XGBClassifier(tree_method="hist", n_jobs=6, random_state=seed, **cfg),
            "random forest (full depth)": RandomForestClassifier(
                n_estimators=100, min_samples_leaf=1, n_jobs=6, random_state=seed),
        }
        for mname, model in models.items():
            model.fit(X.iloc[tr], y[tr], sample_weight=sw)
            pred = model.predict(X.iloc[te])
            for label, m in [("seen in training", seen), ("not seen", ~seen)]:
                rows.append({"protocol": pname, "model": mname, "test rows": label,
                             "share": float(m.mean()),
                             "accuracy": float((pred[m] == yy[m]).mean()),
                             "lookup accuracy": lookup_acc if label == "seen in training" else np.nan})
        print("done:", seed, pname, flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/leakage_check_raw.csv", index=False)
out = res.groupby(["protocol", "model", "test rows"])[["share", "accuracy", "lookup accuracy"]].mean().round(3)
pd.set_option("display.width", 250)
print()
print(out.to_string())