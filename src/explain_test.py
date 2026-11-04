import time
import joblib
import numpy as np
import pandas as pd
import shap
from features import CLASSES

D = "data/processed/"
model = joblib.load("models/classifier_v1.joblib")
Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()

pos = np.random.RandomState(1).choice(len(Xv), 500, replace=False)
S, yS = Xv.iloc[pos], yv[pos]
pred = model.predict(S)

explainer = shap.TreeExplainer(model)

t = time.time()
sv = explainer.shap_values(S)
print(f"500 events explained in {time.time() - t:.1f} s")

t = time.time()
for i in range(20):
    explainer.shap_values(S.iloc[[i]])
print(f"one event: {(time.time() - t) / 20 * 1000:.0f} ms on average")


def values_for(i, k):
    """SHAP values of event i for class k (works for old and new shap versions)."""
    return sv[k][i] if isinstance(sv, list) else sv[i, :, k]


print("\nEXAMPLES (value = scaled feature value, shap > 0 pushes toward the predicted class)")
shown = 0
for i in range(len(S)):
    if pred[i] == yS[i] and pred[i] != 0 and shown < 5:
        v = values_for(i, pred[i])
        top = np.argsort(-np.abs(v))[:5]
        print(f"\nevent {i}: true = {CLASSES[yS[i]]}, predicted = {CLASSES[pred[i]]}")
        for j in top:
            print(f"   {S.columns[j]:<22} value {S.iloc[i, j]:8.3f}   shap {v[j]:+.3f}")
        shown += 1