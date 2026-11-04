import time
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from engine import Engine
from features import CLASSES

D = "data/processed/"
eng = Engine()
eng.iforest.n_jobs = 1
eng.clf.set_params(n_jobs=1)

X = pd.read_parquet(D + "X_val.parquet").sample(200, random_state=1)
explainer = shap.TreeExplainer(eng.clf)
booster = eng.clf.get_booster()
pred = eng.clf.predict(X)


def fast(row_df, k):
    c = booster.predict(xgb.DMatrix(row_df), pred_contribs=True)   # (1, classes, features + 1)
    return c[0, k, :-1]


def slow(row_df, k):
    sv = explainer.shap_values(row_df)
    return sv[k][0] if isinstance(sv, list) else sv[0, :, k]


same_top3, max_diff = [], 0.0
for i in range(len(X)):
    row, k = X.iloc[[i]], int(pred[i])
    a, b = fast(row, k), slow(row, k)
    max_diff = max(max_diff, float(np.abs(a - b).max()))
    same_top3.append(set(np.argsort(-np.abs(a))[:3]) == set(np.argsort(-np.abs(b))[:3]))
print("largest difference between the two methods:", round(max_diff, 6))
print("same top 3 features in", round(float(np.mean(same_top3)) * 100, 1), "% of events")

for name, fn in [("fast (XGBoost built-in)", fast), ("SHAP TreeExplainer", slow)]:
    times = []
    for i in range(3, 60):
        t = time.perf_counter()
        fn(X.iloc[[i]], int(pred[i]))
        times.append((time.perf_counter() - t) * 1000)
    print(f"{name:<26} median {np.median(times):6.1f} ms")