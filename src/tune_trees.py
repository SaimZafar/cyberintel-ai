import json
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

D = "data/processed/"
K = 10
Xf = pd.read_parquet(D + "X_fit.parquet")        # folds 1 to 4
Xv = pd.read_parquet(D + "X_val.parquet")        # fold 0
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
counts = np.bincount(yf, minlength=K)
sw = ((counts.max() / counts) ** 0.5)[yf]
att = yv != 0

rng = np.random.RandomState(7)

lgbm_cfgs = [dict(n_estimators=300, num_leaves=63, learning_rate=0.1, subsample=1.0,
                  colsample_bytree=1.0, min_child_samples=20, reg_lambda=0.0)]
for _ in range(10):
    lgbm_cfgs.append(dict(
        n_estimators=int(rng.choice([200, 300, 400])),
        num_leaves=int(rng.choice([31, 63, 127, 255])),
        learning_rate=float(rng.choice([0.05, 0.1, 0.2])),
        subsample=float(rng.choice([0.7, 0.85, 1.0])),
        colsample_bytree=float(rng.choice([0.6, 0.8, 1.0])),
        min_child_samples=int(rng.choice([10, 20, 50])),
        reg_lambda=float(rng.choice([0.0, 1.0, 5.0, 10.0]))))

rf_cfgs = [dict(n_estimators=200, max_depth=None, min_samples_leaf=2, max_features="sqrt")]
for _ in range(10):
    mf = [("sqrt", "sqrt"), ("0.3", 0.3), ("0.5", 0.5)][rng.randint(3)][1]
    rf_cfgs.append(dict(
        n_estimators=int(rng.choice([200, 300, 400])),
        max_depth=[None, 20, 30][rng.randint(3)],
        min_samples_leaf=int(rng.choice([1, 2, 5])),
        max_features=mf))


def run(kind, cfgs):
    rows, best = [], (-1, None)
    for i, cfg in enumerate(cfgs):
        t0 = time.time()
        if kind == "lgbm":
            m = LGBMClassifier(subsample_freq=1, n_jobs=6, random_state=42, verbosity=-1, **cfg)
        else:
            m = RandomForestClassifier(n_jobs=6, random_state=42, **cfg)
        m.fit(Xf, yf, sample_weight=sw)
        pred = m.predict(Xv)
        f1 = f1_score(yv, pred, average="macro")
        rows.append({"id": i, **{k: str(v) for k, v in cfg.items()}, "macro_f1": f1,
                     "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
                     "normal_recall": float((pred[~att] == 0).mean()), "seconds": time.time() - t0})
        print(f"{kind} config {i}: val macro F1 {f1:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if f1 > best[0]:
            best = (f1, cfg)
        del m
    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    res.to_csv(f"reports/{kind}_tuning.csv", index=False)
    json.dump({"config": best[1], "weight_power": 0.5, "val_macro_f1": best[0]},
              open(f"models/{kind}_best.json", "w"), indent=2)
    pd.set_option("display.width", 250)
    print(res.round(4).head(5).to_string(index=False), "\n", flush=True)


run("lgbm", lgbm_cfgs)
run("rf", rf_cfgs)
print("XGBoost v1 (fold 0): macro F1 0.6567 | folds 1 to 4 in 5-fold CV: 0.6534")