import json
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from features import CLASSES

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()

# weight of each training row: rare classes get a bigger weight
weights = compute_sample_weight("balanced", yf)

# name -> (model, does it use sample weights?)
models = {
    "rf_plain": (RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, n_jobs=8, random_state=42), False),
    "rf_weighted": (RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, n_jobs=8, class_weight="balanced_subsample",
        random_state=42), False),
    "xgb_plain": (XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
        n_jobs=8, random_state=42), False),
    "xgb_weighted": (XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
        n_jobs=8, random_state=42), True),
}

rows, trained = [], {}
for name, (model, use_weights) in models.items():
    start = time.time()
    if use_weights:
        model.fit(Xf, yf, sample_weight=weights)
    else:
        model.fit(Xf, yf)
    pred = model.predict(Xv)
    row = {
        "model": name,
        "accuracy": accuracy_score(yv, pred),
        "macro_f1": f1_score(yv, pred, average="macro"),
        "weighted_f1": f1_score(yv, pred, average="weighted"),
        "binary_attack_f1": f1_score((yv != 0).astype(int), (pred != 0).astype(int)),
        "seconds": time.time() - start,
    }
    rows.append(row)
    trained[name] = (model, pred)
    print({k: (v if k == "model" else round(v, 4)) for k, v in row.items()})

results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
results.to_csv("reports/model_comparison.csv", index=False)
print("\n", results.round(4).to_string(index=False))

# keep the best model by macro F1 and show its per-class results
best = results.iloc[0]["model"]
model, pred = trained[best]
print("\nBEST:", best)
print(classification_report(yv, pred, labels=list(range(len(CLASSES))),
                            target_names=CLASSES, zero_division=0, digits=3))
joblib.dump(model, f"models/best_{best}.joblib", compress=3)
with open("models/best_model.json", "w") as f:
    json.dump({"name": best}, f)