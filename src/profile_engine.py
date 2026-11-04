import json
import time
import numpy as np
import pandas as pd
import shap
from engine import Engine
from features import make_features

D = "data/processed/"
eng = Engine()
cfg = json.load(open("models/feature_config.json"))
cats, names = cfg["cats"], cfg["feature_names"]
explainer = shap.TreeExplainer(eng.clf)

raw = pd.read_parquet(D + "demo_pool.parquet")
raw = raw.sample(60, random_state=0).drop(columns=["attack_cat", "label", "group"])


def timed(fn):
    a = time.perf_counter()
    fn()
    return (time.perf_counter() - a) * 1000


def measure(label):
    t = {"make features": [], "classifier": [], "anomaly score": [], "SHAP": []}
    for i in range(len(raw)):
        r = raw.iloc[[i]]
        X = make_features(r, cats)[names]
        if i < 3:                      # warm-up, not counted
            eng.clf.predict_proba(X); eng.anomaly(X); explainer.shap_values(X)
            continue
        t["make features"].append(timed(lambda: make_features(r, cats)[names]))
        t["classifier"].append(timed(lambda: eng.clf.predict_proba(X)))
        t["anomaly score"].append(timed(lambda: eng.anomaly(X)))
        t["SHAP"].append(timed(lambda: explainer.shap_values(X)))
    print(f"\n{label}  (median ms per event)")
    total = 0
    for k, v in t.items():
        print(f"  {k:<15}{np.median(v):8.1f}")
        total += np.median(v)
    print(f"  {'total':<15}{total:8.1f}")


measure("current settings")

eng.iforest.n_jobs = 1
eng.clf.set_params(n_jobs=1)
measure("one thread for the forest and XGBoost")