import numpy as np
import pandas as pd
from engine import Engine
from features import CLASSES

D = "data/processed/"
eng = Engine()
X = pd.read_parquet(D + "X_test_eval.parquet")
y = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
g = pd.read_parquet(D + "test_eval.parquet")["group"].astype("uint64")

s = 1.0 - eng.clf.predict_proba(X)[:, 0]
dev, fin = (g % 2 == 0).to_numpy(), (g % 2 == 1).to_numpy()

for budget in [10, 20, 50]:
    t = float(np.quantile(s[dev & (y == 0)], 1 - budget / 1000))
    alert = s[fin] > t
    true = pd.Series([CLASSES[k] for k in y[fin]])
    share = pd.Series(alert).groupby(true).mean().reindex(CLASSES).round(3)
    print(f"\nBUDGET {budget} per 1000 (threshold {t:.3f}): share of each true class that alerts")
    print(share.to_string())