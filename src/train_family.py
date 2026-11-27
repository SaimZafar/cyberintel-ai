import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

D = "data/processed/"
NAMES = ["Normal", "Generic", "Exploit-type", "Fuzzers", "Reconnaissance", "Shellcode", "Worms"]
MERGE = {4: 2, 6: 2, 7: 2}                 # DoS, Analysis, Backdoor -> Exploits
KEEP = [0, 1, 2, 3, 5, 8, 9]               # old class numbers that remain
NEW = {old: i for i, old in enumerate(KEEP)}


def fam(y):
    return np.array([NEW[MERGE.get(v, v)] for v in y])


Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
Xt = pd.read_parquet(D + "X_test_eval.parquet")
yf_old = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv_old = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
yt_old = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
yf, yv, yt = fam(yf_old), fam(yv_old), fam(yt_old)

cfg = json.load(open("models/classifier_v1.json"))["config"]
counts = np.bincount(yf, minlength=len(NAMES))

# 1) choose the class-weight strength on VALIDATION only
best = None
for p in [0.25, 0.5, 0.75]:
    w = ((counts.max() / counts) ** p)[yf]
    m = XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cfg)
    m.fit(Xf, yf, sample_weight=w)
    score = f1_score(yv, m.predict(Xv), average="macro")
    print(f"weight power {p}: validation macro F1 {score:.4f}", flush=True)
    if best is None or score > best[0]:
        best = (score, p, m)
val_f1, power, model = best
print(f"\nchosen power {power} (validation macro F1 {val_f1:.4f})")
joblib.dump(model, "models/classifier_family_v1.joblib", compress=3)

# 2) the test set is looked at ONCE, after the choice above
pred = model.predict(Xt)
old_model = joblib.load("models/classifier_v1.joblib")
old_merged = fam(old_model.predict(Xt))

print("\nOFFICIAL TEST (same rows, 7-family view)")
print(f"{'':<28}{'accuracy':>10}{'macro F1':>10}")
print(f"{'10-class model, merged':<28}{accuracy_score(yt, old_merged):10.3f}{f1_score(yt, old_merged, average='macro'):10.3f}")
print(f"{'7-family model (new)':<28}{accuracy_score(yt, pred):10.3f}{f1_score(yt, pred, average='macro'):10.3f}")
print("\nper family F1 (old merged vs new):")
a = f1_score(yt, old_merged, average=None)
b = f1_score(yt, pred, average=None)
for n, x, y in zip(NAMES, a, b):
    print(f"  {n:<16}{x:7.3f}{y:7.3f}")