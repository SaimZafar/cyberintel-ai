import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

D = "data/processed/"
K = 10
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
Xt = pd.read_parquet(D + "X_test_eval.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
yt = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
cfg = json.load(open("models/classifier_v1.json"))
cols = list(Xf.columns)
CAT = ("proto_", "service_", "state_")

# Zoghi and Serpen (2024), Table 5: 24 features chosen by Elastic Net, 8 of them by SFS
Z24_NUM = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss",
           "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt"]
Z8_NUM = ["sbytes", "rate", "dload", "sjit", "djit", "tcprtt"]       # plus proto and service
missing = [c for c in Z24_NUM if c not in cols]
print("Zoghi numeric features not found in our columns:", missing)

cat_all = [c for c in cols if c.startswith(CAT)]
z24 = [c for c in Z24_NUM if c in cols] + cat_all
z8 = [c for c in Z8_NUM if c in cols] + [c for c in cols if c.startswith(("proto_", "service_"))]
ct = [c for c in cols if c.startswith("ct_")]
timing = [c for c in ["synack", "ackdat"] if c in cols]
sizes = [c for c in ["smean", "dmean", "trans_depth", "response_body_len"] if c in cols]
rest = [c for c in cols if c not in z24 and c not in ct and c not in timing and c not in sizes]
print("features excluded by Zoghi that are not ct_*, timing or size groups:", rest)

sets = {
    "all features (reference)": cols,
    "Zoghi 24 features": z24,
    "Zoghi 8 features (SFS)": z8,
    "Zoghi 24 + ct_*": z24 + ct,
    "Zoghi 24 + synack, ackdat": z24 + timing,
    "Zoghi 24 + sizes (smean, dmean, ...)": z24 + sizes,
    "Zoghi 24 + remaining flags": z24 + rest,
}
counts = np.bincount(yf, minlength=K)
sw = ((counts.max() / counts) ** cfg["weight_power"])[yf]


def scores(y, proba):
    pred = proba.argmax(1)
    att = y != 0
    return (f1_score(y, pred, average="macro"),
            f1_score(att.astype(int), (pred != 0).astype(int)),
            float((pred[~att] == 0).mean()))


rows = []
for name, keep in sets.items():
    keep = list(dict.fromkeys(keep))
    m = XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cfg["config"])
    m.fit(Xf[keep], yf, sample_weight=sw)
    pv, pt = m.predict_proba(Xv[keep]), m.predict_proba(Xt[keep])
    vm, vb, vn = scores(yv, pv)
    tm, tb, tn = scores(yt, pt)
    # threshold for 20 false alerts per 1000 Normal, chosen on the validation fold only
    av, at = 1 - pv[:, 0], 1 - pt[:, 0]
    thr = float(np.quantile(av[yv == 0], 0.98))
    rows.append({"feature set": name, "n": len(keep),
                 "val_macroF1": vm, "test_macroF1": tm, "val_NormalRecall": vn, "test_NormalRecall": tn,
                 "test_binaryF1": tb, "val_threshold": thr,
                 "test_FA_per_1000": 1000 * float((at[yt == 0] > thr).mean()),
                 "test_attack_recall": float((at[yt != 0] > thr).mean())})
    print("done:", name, flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/zoghi_features.csv", index=False)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)
print()
print(res.round(3).to_string(index=False))