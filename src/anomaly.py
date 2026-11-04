import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from features import CLASSES

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
yv_bin = pd.read_parquet(D + "y_val.parquet")["y_bin"].to_numpy()

# the detector only ever sees NORMAL traffic (class 0)
normal_fit = Xf[yf == 0].sample(frac=1.0, random_state=42)   # shuffled
cut = int(len(normal_fit) * 0.7)
train_part, ref_part = normal_fit.iloc[:cut], normal_fit.iloc[cut:]

iforest = IsolationForest(n_estimators=300, random_state=42, n_jobs=8)
iforest.fit(train_part)

# reference: how normal rows that the forest has NOT seen score (higher = more normal)
ref = np.sort(iforest.score_samples(ref_part))


def anomaly_score(X):
    """0 to 1: share of normal reference rows that look MORE normal than this event."""
    s = iforest.score_samples(X)
    more_normal = len(ref) - np.searchsorted(ref, s, side="right")
    return more_normal / len(ref)


a_val = anomaly_score(Xv)
THRESHOLD = 0.95   # events above this are treated as "unusual"

print("AUROC (normal vs attack):", round(roc_auc_score(yv_bin, a_val), 4))
print(f"false alarm rate on normal traffic (score > {THRESHOLD}):",
      round(float((a_val[yv == 0] > THRESHOLD).mean()), 4))
print(f"\n{'class':<16}{'flagged':>9}{'mean score':>12}")
for i, c in enumerate(CLASSES):
    m = yv == i
    print(f"{c:<16}{(a_val[m] > THRESHOLD).mean():9.3f}{a_val[m].mean():12.3f}")

joblib.dump(iforest, "models/anomaly_iforest.joblib", compress=3)
np.save("models/anomaly_reference.npy", ref)
with open("models/anomaly_config.json", "w") as f:
    json.dump({"threshold": THRESHOLD, "n_reference": int(len(ref))}, f)
print("\nsaved models/anomaly_iforest.joblib")