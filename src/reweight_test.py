import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
Xt = pd.read_parquet(D + "X_test_eval.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
yt = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
gt = pd.read_parquet(D + "test_eval.parquet")["group"].astype("uint64")

part = (gt % 4).to_numpy()
adapt = part == 0       # unlabeled features only, used to estimate the shift
evalm = part == 2       # scored with labels
# parts 1 and 3 stay untouched
Xa, Xe, ye = Xt[adapt], Xt[evalm], yt[evalm]
print("adapt rows:", len(Xa), "| eval rows:", len(ye))

# cross-fitted estimate of how test-like each training row is
ratio = np.zeros(len(Xf))
for tr, ho in KFold(5, shuffle=True, random_state=0).split(Xf):
    X = pd.concat([Xf.iloc[tr], Xa])
    d = np.r_[np.zeros(len(tr)), np.ones(len(Xa))]
    m = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                      tree_method="hist", n_jobs=6, random_state=0).fit(X, d)
    p = np.clip(m.predict_proba(Xf.iloc[ho])[:, 1], 1e-3, 1 - 1e-3)
    ratio[ho] = p / (1 - p) * (len(tr) / len(Xa))     # prior correction
print("raw weight ratio: median %.2f, 95th pct %.2f, max %.1f"
      % (np.median(ratio), np.percentile(ratio, 95), ratio.max()))

cfg = json.load(open("models/classifier_v1.json"))["config"]
counts = np.bincount(yf, minlength=10)
class_w = ((counts.max() / counts) ** 0.5)[yf]


def score(y, pred):
    att = y != 0
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
            "normal_recall": float((pred[~att] == 0).mean())}


rows = []
for power in [0.0, 0.5, 1.0]:
    w = np.clip(ratio, 0.05, 20) ** power
    w = w / w.mean()
    model = XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cfg)
    model.fit(Xf, yf, sample_weight=class_w * w)
    r = {"shift power": power}
    for label, X, y in [("val", Xv, yv), ("test_eval", Xe, ye)]:
        for k, v in score(y, model.predict(X)).items():
            r[f"{label}_{k}"] = v
    rows.append(r)
    print("done: power", power, flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/reweight_test.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print()
print(res.round(3).to_string(index=False))