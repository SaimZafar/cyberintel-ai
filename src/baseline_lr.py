import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
from features import CLASSES

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")
yv = pd.read_parquet(D + "y_val.parquet")

# logistic regression needs scaled inputs (fit the scaler on the training part only)
scaler = StandardScaler().fit(Xf)
model = LogisticRegression(max_iter=500, class_weight="balanced")
model.fit(scaler.transform(Xf), yf["y"])

pred = model.predict(scaler.transform(Xv))
true = yv["y"].to_numpy()

# binary view: anything that is not Normal counts as an attack
pred_attack = (pred != 0).astype(int)
true_attack = yv["y_bin"].to_numpy()

results = {
    "accuracy": accuracy_score(true, pred),
    "macro_f1": f1_score(true, pred, average="macro"),
    "weighted_f1": f1_score(true, pred, average="weighted"),
    "binary_attack_f1": f1_score(true_attack, pred_attack),
}
print({k: round(v, 4) for k, v in results.items()})
print()
print(classification_report(true, pred, labels=list(range(len(CLASSES))),
                            target_names=CLASSES, zero_division=0, digits=3))

joblib.dump({"scaler": scaler, "model": model}, "models/baseline_lr.joblib")
with open("reports/baseline_lr.json", "w") as f:
    json.dump(results, f, indent=2)