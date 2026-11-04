import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from features import CLASSES

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
fit_groups = pd.read_parquet(D + "fit.parquet")["group"]
val_groups = pd.read_parquet(D + "val.parquet")["group"]

keep_fit = ~fit_groups.duplicated().to_numpy()   # first copy of each row only
keep_val = ~val_groups.duplicated().to_numpy()

idx = {c: i for i, c in enumerate(CLASSES)}
labels = list(range(len(CLASSES)))
P = 0.5  # weight strength chosen in the last step

rows = []
for name, mask in [("all rows", np.ones(len(yf), dtype=bool)), ("deduplicated", keep_fit)]:
    X, y = Xf[mask], yf[mask]
    counts = np.bincount(y, minlength=len(CLASSES))
    class_weight = (counts.max() / counts) ** P
    model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                          tree_method="hist", n_jobs=8, random_state=42)
    model.fit(X, y, sample_weight=class_weight[y])
    pred = model.predict(Xv)

    f1s = f1_score(yv, pred, labels=labels, average=None, zero_division=0)
    rows.append({
        "training": name,
        "train_rows": len(y),
        "macro_f1_val": f1_score(yv, pred, average="macro"),
        "macro_f1_val_dedup": f1_score(yv[keep_val], pred[keep_val], average="macro"),
        "binary_f1": f1_score((yv != 0).astype(int), (pred != 0).astype(int)),
        "f1_Normal": f1s[idx["Normal"]],
        "f1_DoS": f1s[idx["DoS"]],
        "f1_Backdoor": f1s[idx["Backdoor"]],
        "f1_Worms": f1s[idx["Worms"]],
    })
    print("done:", name)

result = pd.DataFrame(rows)
result.to_csv("reports/dedup_test.csv", index=False)
print()
print(result.round(3).to_string(index=False))