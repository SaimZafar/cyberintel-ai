import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()

P = 0.5
counts = np.bincount(yf, minlength=10)
class_weight = (counts.max() / counts) ** P
sample_weight = class_weight[yf]

# config 0 = the setting we already use, so we can see if tuning really helps
configs = [dict(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=1.0,
                colsample_bytree=1.0, min_child_weight=1, reg_lambda=1)]

rng = np.random.RandomState(7)
for _ in range(10):
    configs.append(dict(
        n_estimators=int(rng.choice([200, 300, 400])),
        max_depth=int(rng.choice([4, 6, 8, 10])),
        learning_rate=float(rng.choice([0.05, 0.1, 0.2])),
        subsample=float(rng.choice([0.7, 0.85, 1.0])),
        colsample_bytree=float(rng.choice([0.6, 0.8, 1.0])),
        min_child_weight=int(rng.choice([1, 3, 5])),
        reg_lambda=float(rng.choice([1, 5, 10])),
    ))

rows, best_score, best_model, best_cfg = [], -1, None, None
for i, cfg in enumerate(configs):
    start = time.time()
    model = XGBClassifier(tree_method="hist", n_jobs=8, random_state=42, **cfg)
    model.fit(Xf, yf, sample_weight=sample_weight)
    pred = model.predict(Xv)
    score = f1_score(yv, pred, average="macro")
    rows.append({"id": i, **cfg, "macro_f1": score,
                 "binary_f1": f1_score((yv != 0).astype(int), (pred != 0).astype(int)),
                 "seconds": time.time() - start})
    print(f"config {i}: macro_f1 = {score:.4f}  ({time.time() - start:.0f}s)")
    if score > best_score:
        best_score, best_model, best_cfg = score, model, cfg

result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
result.to_csv("reports/xgb_tuning.csv", index=False)
print()
print(result.round(4).head(6).to_string(index=False))

joblib.dump(best_model, "models/classifier_v1.joblib", compress=3)
with open("models/classifier_v1.json", "w") as f:
    json.dump({"config": best_cfg, "weight_power": P, "val_macro_f1": best_score}, f, indent=2)
print("\nsaved models/classifier_v1.joblib with", best_cfg)