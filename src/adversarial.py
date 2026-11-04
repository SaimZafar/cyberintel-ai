import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from features import CLASSES

D = "data/processed/"
Xtr = pd.read_parquet(D + "X_fit.parquet")
Xte = pd.read_parquet(D + "X_test_eval.parquet")
gtr = pd.read_parquet(D + "fit.parquet")["group"]
gte = pd.read_parquet(D + "test_eval.parquet")["group"]
ytr = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yte = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()

# one copy of each pattern per set, and only patterns that exist in one set only
keep_tr = ~gtr.duplicated().to_numpy() & ~gtr.isin(set(gte)).to_numpy()
keep_te = ~gte.duplicated().to_numpy() & ~gte.isin(set(gtr)).to_numpy()
Xtr, ytr = Xtr[keep_tr], ytr[keep_tr]
Xte, yte = Xte[keep_te], yte[keep_te]
print("distinct patterns kept: train", len(Xtr), "| test", len(Xte))


def run(a, b, label, n=20000, seed=0):
    """Can a model tell rows of a (label 0) from rows of b (label 1)?"""
    a = a.sample(min(n, len(a)), random_state=seed)
    b = b.sample(min(n, len(b)), random_state=seed)
    X = pd.concat([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          tree_method="hist", n_jobs=6, random_state=42)
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    model.fit(X, y)
    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print(f"\n{label}: AUC = {auc:.3f}  (0.5 = cannot tell them apart)")
    print("  most useful features for telling them apart:")
    for name, v in imp.head(8).items():
        print(f"    {name:<22}{v:8.1f}")
    return auc


run(Xtr, Xte, "ALL rows, train vs test")
run(Xtr[ytr == 0], Xte[yte == 0], "NORMAL rows only, train vs test")
for k in [CLASSES.index(c) for c in ["Fuzzers", "Exploits", "DoS"]]:
    if (ytr == k).sum() > 500 and (yte == k).sum() > 500:
        run(Xtr[ytr == k], Xte[yte == k], f"{CLASSES[k]} rows only, train vs test")