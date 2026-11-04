import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

D = "data/processed/"
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
Xt = pd.read_parquet(D + "X_test_eval.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
yt = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
gt = pd.read_parquet(D + "test_eval.parquet")["group"].astype("uint64")

dev = (gt % 2 == 0).to_numpy()          # dev half; the other half is not touched here
Xd, yd = Xt[dev], yt[dev]
print("test dev half:", len(yd), "rows")

cfg = json.load(open("models/classifier_v1.json"))["config"]
counts = np.bincount(yf, minlength=10)
sw = ((counts.max() / counts) ** 0.5)[yf]

cols = list(Xf.columns)
ttl = ["sttl", "dttl", "ct_state_ttl"]
ct = [c for c in cols if c.startswith("ct_") and c not in ("ct_ftp_cmd", "ct_flw_http_mthd")]
added8 = ["sttl", "dttl", "ct_state_ttl", "ct_srv_src", "ct_dst_ltm",
          "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"]
sets = {
    "all 60 features": [],
    "drop the 8 added features": added8,
    "drop TTL features only": ttl,
    "drop all ct_* only": ct,
    "drop TTL and all ct_*": sorted(set(ttl + ct)),
}


def score(y, pred):
    att = y != 0
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
        "normal_recall": float((pred[~att] == 0).mean()),
    }


rows = []
for name, drop in sets.items():
    keep = [c for c in cols if c not in drop]
    model = XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cfg)
    model.fit(Xf[keep], yf, sample_weight=sw)
    r = {"feature set": name, "n_features": len(keep)}
    for label, X, y in [("val", Xv, yv), ("test_dev", Xd, yd)]:
        for k, v in score(y, model.predict(X[keep])).items():
            r[f"{label}_{k}"] = v
    rows.append(r)
    print("done:", name, flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/shift_ablation.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print()
print(res.round(3).to_string(index=False))