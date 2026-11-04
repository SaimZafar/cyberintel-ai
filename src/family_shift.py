import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
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
cols = list(Xf.columns)

Z24_NUM = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload", "sloss",
           "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt"]
z24 = Z24_NUM + [c for c in cols if c.startswith(("proto_", "service_", "state_"))]
featsets = {"all 60": cols, "Zoghi 24": z24}

counts = np.bincount(yf, minlength=K)
cx = json.load(open("models/classifier_v1.json"))
cl = json.load(open("models/lgbm_best.json"))
cr = json.load(open("models/rf_best.json"))


def make(name):
    if name == "xgb":
        return XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **cx["config"]), cx["weight_power"]
    if name == "lgbm":
        return LGBMClassifier(subsample_freq=1, n_jobs=6, random_state=42, verbosity=-1, **cl["config"]), cl["weight_power"]
    return RandomForestClassifier(n_jobs=6, random_state=42, **cr["config"]), cr["weight_power"]


rows = []
for name in ["xgb", "lgbm", "rf"]:
    for fname, keep in featsets.items():
        m, power = make(name)
        m.fit(Xf[keep], yf, sample_weight=((counts.max() / counts) ** power)[yf])
        pv, pt = m.predict_proba(Xv[keep]), m.predict_proba(Xt[keep])
        pred = pt.argmax(1)
        att = yt != 0
        av, at = 1 - pv[:, 0], 1 - pt[:, 0]
        thr = float(np.quantile(av[yv == 0], 0.98))               # validation-chosen, budget 20
        thr_oracle = float(np.quantile(at[~att], 0.98))           # DIAGNOSTIC: uses test labels
        rows.append({"model": name, "features": fname,
                     "val_macroF1": f1_score(yv, pv.argmax(1), average="macro"),
                     "test_macroF1": f1_score(yt, pred, average="macro"),
                     "test_NormalRecall": float((pred[~att] == 0).mean()),
                     "test_binaryF1": f1_score(att.astype(int), (pred != 0).astype(int)),
                     "FA_at_val_thr": 1000 * float((at[~att] > thr).mean()),
                     "recall_at_val_thr": float((at[att] > thr).mean()),
                     "oracle_recall_at_FA20": float((at[att] > thr_oracle).mean())})
        print("done:", name, fname, flush=True)

res = pd.DataFrame(rows)
res.to_csv("reports/family_shift.csv", index=False)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)
print()
print(res.round(3).to_string(index=False))