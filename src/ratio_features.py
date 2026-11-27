import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

D = "data/processed/"
K = 10
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
cfg = json.load(open("models/classifier_v1.json"))["config"]
power = json.load(open("models/classifier_v1.json"))["weight_power"]

# Most columns are already log(1+x), so a difference of two columns is a log ratio.
PAIRS = {
    "s_bytes_per_pkt": ("sbytes", "spkts"),
    "d_bytes_per_pkt": ("dbytes", "dpkts"),
    "byte_ratio": ("sbytes", "dbytes"),
    "pkt_ratio": ("spkts", "dpkts"),
    "load_ratio": ("sload", "dload"),
    "s_loss_rate": ("sloss", "spkts"),
    "d_loss_rate": ("dloss", "dpkts"),
    "jitter_ratio": ("sjit", "djit"),
    "inpkt_ratio": ("sinpkt", "dinpkt"),
    "mean_ratio": ("smean", "dmean"),
    "handshake_ratio": ("synack", "ackdat"),
    "ttl_diff": ("sttl", "dttl"),          # TTL columns are plain numbers, so this is a plain difference
}
missing = sorted({c for pair in PAIRS.values() for c in pair if c not in Xf.columns})
print("columns not found:", missing)
PAIRS = {k: v for k, v in PAIRS.items() if v[0] in Xf.columns and v[1] in Xf.columns}


def add_ratios(X):
    X = X.copy()
    for name, (a, b) in PAIRS.items():
        X[name] = (X[a] - X[b]).astype("float32")
    return X


Xf2, Xv2 = add_ratios(Xf), add_ratios(Xv)
counts = np.bincount(yf, minlength=K)
sw = ((counts.max() / counts) ** power)[yf]

rows = []
for name, (A, B) in {"current 60 features": (Xf, Xv),
                     f"+ {len(PAIRS)} ratio features": (Xf2, Xv2)}.items():
    for seed in [42, 7]:
        m = XGBClassifier(tree_method="hist", n_jobs=6, random_state=seed, **cfg)
        m.fit(A, yf, sample_weight=sw)
        p = m.predict(B)
        f1 = f1_score(yv, p, average=None, labels=list(range(K)), zero_division=0)
        rows.append({"features": name, "seed": seed,
                     "val_macro_F1": f1.mean(),
                     "Normal_recall": recall_score(yv == 0, p == 0),
                     "F1_Fuzzers": f1[3], "F1_Shellcode": f1[8], "F1_Exploits": f1[2]})
        print("done:", name, "seed", seed, flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/ratio_features.csv", index=False)
pd.set_option("display.width", 200)
print()
print(res.round(4).to_string(index=False))
print()
print(res.groupby("features")[["val_macro_F1", "Normal_recall", "F1_Fuzzers", "F1_Shellcode"]].mean().round(4))