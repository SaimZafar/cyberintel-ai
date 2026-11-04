import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from features import CLASSES

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()

counts = np.bincount(yf, minlength=len(CLASSES))
sw = ((counts.max() / counts) ** 0.5)[yf]          # same weights as classifier v1

probs = {"xgb": joblib.load("models/classifier_v1.joblib").predict_proba(Xv)}

others = {
    "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=6, random_state=42),
    "extratrees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=6, random_state=42),
    "lgbm": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.85,
                           subsample_freq=1, colsample_bytree=0.6, n_jobs=6,
                           random_state=42, verbosity=-1),
}
for name, m in others.items():
    m.fit(Xf, yf, sample_weight=sw)
    probs[name] = m.predict_proba(Xv)
    print("trained", name, flush=True)

idx = {c: i for i, c in enumerate(CLASSES)}
labels = list(range(len(CLASSES)))
combos = {k: [k] for k in probs}
combos["xgb+lgbm"] = ["xgb", "lgbm"]
combos["xgb+rf"] = ["xgb", "rf"]
combos["xgb+extratrees"] = ["xgb", "extratrees"]
combos["all four"] = list(probs)

rows = []
for name, members in combos.items():
    p = np.mean([probs[m] for m in members], axis=0)
    pred = p.argmax(axis=1)
    f1s = f1_score(yv, pred, labels=labels, average=None, zero_division=0)
    rows.append({"model": name,
                 "macro_f1": f1_score(yv, pred, average="macro"),
                 "binary_f1": f1_score((yv != 0).astype(int), (pred != 0).astype(int)),
                 "f1_DoS": f1s[idx["DoS"]], "f1_Analysis": f1s[idx["Analysis"]],
                 "f1_Backdoor": f1s[idx["Backdoor"]], "f1_Worms": f1s[idx["Worms"]]})
result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
result.to_csv("reports/ensemble_test.csv", index=False)
print()
print(result.round(4).to_string(index=False))