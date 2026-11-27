import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

D = "data/processed/"
clf = joblib.load("models/classifier_v1.joblib")
X = pd.read_parquet(D + "X_test_eval.parquet")
y = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
pred = clf.predict(X)

# class numbers: 2 Exploits, 4 DoS, 6 Analysis, 7 Backdoor -> merged into 2
merge = {4: 2, 6: 2, 7: 2}
fy = np.array([merge.get(v, v) for v in y])
fp = np.array([merge.get(v, v) for v in pred])

print("10-class : accuracy %.3f  macro F1 %.3f" % (accuracy_score(y, pred), f1_score(y, pred, average="macro")))
print("7-family : accuracy %.3f  macro F1 %.3f" % (accuracy_score(fy, fp), f1_score(fy, fp, average="macro")))
print("per family F1:", np.round(f1_score(fy, fp, average=None), 3))