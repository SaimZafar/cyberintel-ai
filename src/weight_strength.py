import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score
from xgboost import XGBClassifier
from features import CLASSES

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()

counts = np.bincount(yf, minlength=len(CLASSES))
labels = list(range(len(CLASSES)))
idx = {c: i for i, c in enumerate(CLASSES)}

rows = []
for p in [0, 0.25, 0.5, 0.75, 1.0]:
    class_weight = (counts.max() / counts) ** p
    model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                          tree_method="hist", n_jobs=8, random_state=42)
    model.fit(Xf, yf, sample_weight=class_weight[yf])
    pred = model.predict(Xv)

    f1s = f1_score(yv, pred, labels=labels, average=None, zero_division=0)
    prec = precision_score(yv, pred, labels=labels, average=None, zero_division=0)
    row = {
        "power": p,
        "macro_f1": f1_score(yv, pred, average="macro"),
        "accuracy": (pred == yv).mean(),
        "binary_f1": f1_score((yv != 0).astype(int), (pred != 0).astype(int)),
        "backdoor_prec": prec[idx["Backdoor"]],
        "f1_DoS": f1s[idx["DoS"]],
        "f1_Analysis": f1s[idx["Analysis"]],
        "f1_Backdoor": f1s[idx["Backdoor"]],
        "f1_Shellcode": f1s[idx["Shellcode"]],
        "f1_Worms": f1s[idx["Worms"]],
    }
    rows.append(row)
    print("done p =", p, "macro_f1 =", round(row["macro_f1"], 4))

result = pd.DataFrame(rows)
result.to_csv("reports/weight_strength.csv", index=False)
print()
print(result.round(3).to_string(index=False))