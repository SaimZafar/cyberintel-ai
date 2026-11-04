import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
name = sys.argv[1] if len(sys.argv) > 1 else "xgb"
Path("reports/cv").mkdir(parents=True, exist_ok=True)

# ---- data: official training partition; folds from the verified flags file ----
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
fit_raw = pd.read_parquet(D + "fit.parquet")
val_raw = pd.read_parquet(D + "val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
X = pd.concat([Xf, Xv], ignore_index=True)
y = np.r_[yf, yv]
ids = np.r_[fit_raw["id"].to_numpy(), val_raw["id"].to_numpy()]
groups = np.r_[fit_raw["group"].to_numpy(), val_raw["group"].to_numpy()]
flags = pd.read_csv("reports/release/unsw_nb15_row_flags.csv", dtype={"group": str},
                    usecols=["partition", "id", "train_fold_group_aware"])
flags = flags[flags["partition"] == "train"].set_index("id")["train_fold_group_aware"]
fold = flags.loc[ids].to_numpy()
assert len(X) == 175341 and (fold >= 0).all(), "fold assignment problem"
print(name, "| rows per fold:", np.bincount(fold).tolist(), flush=True)


def inner_split(tr_idx):
    """Group-aware 90/10 split of the training folds, used only for early stopping."""
    codes = pd.factorize(groups[tr_idx])[0]
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    a, b = next(sgkf.split(np.zeros(len(tr_idx)), y[tr_idx], groups=codes))
    a, b = tr_idx[a], tr_idx[b]
    assert not set(groups[a]) & set(groups[b])
    return a, b


def mlp_fit_predict(tr, te):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    c = json.load(open("models/mlp_best.json"))["config"]
    a, b = inner_split(tr)
    sc = StandardScaler().fit(X.iloc[a])                      # training part only
    T = lambda idx: torch.tensor(sc.transform(X.iloc[idx]), dtype=torch.float32)
    Aa, Ab, Ae = T(a), T(b), T(te)
    ya, yb = torch.tensor(y[a], dtype=torch.long), y[b]
    torch.manual_seed(0)
    layers, d = [], Aa.shape[1]
    for h in c["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(c["dropout"])]
        d = h
    model = nn.Sequential(*layers, nn.Linear(d, K))
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    cnt = np.bincount(y[a], minlength=K)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor((cnt.max() / cnt) ** c["power"], dtype=torch.float32))
    best, state, bad, n = -1.0, None, 0, len(ya)
    for epoch in range(40):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, c["batch"]):
            idx = perm[i:i + c["batch"]]
            opt.zero_grad()
            loss_fn(model(Aa[idx]), ya[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f1 = f1_score(yb, model(Ab).argmax(1).numpy(), average="macro")
        if f1 > best:
            best, bad = f1, 0
            state = {k_: v.clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(Ae), dim=1).numpy()


def tabnet_fit_predict(tr, te):
    import torch
    from pytorch_tabnet.metrics import Metric
    from pytorch_tabnet.tab_model import TabNetClassifier
    c = json.load(open("models/tabnet_best.json"))["config"]
    a, b = inner_split(tr)
    f32 = lambda idx: X.iloc[idx].to_numpy(dtype=np.float32)
    cnt = np.bincount(y[a], minlength=K)
    wdict = {i: float((cnt.max() / cnt[i]) ** 0.5) for i in range(K)}

    class MacroF1(Metric):
        def __init__(self):
            self._name = "macro_f1"
            self._maximize = True

        def __call__(self, y_true, y_score):
            return f1_score(y_true, np.argmax(y_score, axis=1), average="macro")

    torch.manual_seed(0)
    np.random.seed(0)
    clf = TabNetClassifier(n_d=int(c["n_d"]), n_a=int(c["n_a"]), n_steps=int(c["n_steps"]),
                           gamma=float(c["gamma"]), lambda_sparse=float(c["lambda_sparse"]),
                           seed=0, device_name="cpu", verbose=0,
                           optimizer_params=dict(lr=float(c["lr"])))
    clf.fit(f32(a), y[a], eval_set=[(f32(b), y[b])], eval_name=["val"], eval_metric=[MacroF1],
            max_epochs=25, patience=5, batch_size=int(c["batch"]), virtual_batch_size=128,
            weights=wdict, drop_last=False)
    return clf.predict_proba(f32(te))


def fit_predict(name, tr, te):
    cnt = np.bincount(y[tr], minlength=K)
    if name == "xgb":
        from xgboost import XGBClassifier
        c = json.load(open("models/classifier_v1.json"))
        w = ((cnt.max() / cnt) ** c["weight_power"])[y[tr]]
        m = XGBClassifier(tree_method="hist", n_jobs=6, random_state=42, **c["config"])
        m.fit(X.iloc[tr], y[tr], sample_weight=w)
        return m.predict_proba(X.iloc[te])
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        c = json.load(open("models/lgbm_best.json"))
        w = ((cnt.max() / cnt) ** c["weight_power"])[y[tr]]
        m = LGBMClassifier(subsample_freq=1, n_jobs=6, random_state=42, verbosity=-1, **c["config"])
        m.fit(X.iloc[tr], y[tr], sample_weight=w)
        return m.predict_proba(X.iloc[te])
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        c = json.load(open("models/rf_best.json"))
        w = ((cnt.max() / cnt) ** c["weight_power"])[y[tr]]
        m = RandomForestClassifier(n_jobs=6, random_state=42, **c["config"])
        m.fit(X.iloc[tr], y[tr], sample_weight=w)
        return m.predict_proba(X.iloc[te])
    if name == "mlp":
        return mlp_fit_predict(tr, te)
    if name == "tabnet":
        return tabnet_fit_predict(tr, te)
    raise SystemExit(f"unknown model '{name}'")


def evaluate(yt, pred):
    att = yt != 0
    sup = np.bincount(yt, minlength=K)
    f1s = f1_score(yt, pred, labels=list(range(K)), average=None, zero_division=0)
    rec = np.array([(pred[yt == k] == k).mean() if sup[k] else np.nan for k in range(K)])
    r = {"accuracy": accuracy_score(yt, pred), "macro_f1": f1s[sup > 0].mean(),
         "weighted_f1": f1_score(yt, pred, average="weighted"),
         "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
         "normal_recall": float((pred[~att] == 0).mean())}
    r.update({f"recall_{CLASSES[k]}": rec[k] for k in range(K)})
    r.update({f"f1_{CLASSES[k]}": f1s[k] for k in range(K)})
    return r


rows = []
oof = np.full(len(y), -1, dtype=np.int8)
oof_p = np.zeros((len(y), K), dtype=np.float32)
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    assert not set(groups[tr]) & set(groups[te]), "identical rows on both sides of a fold"
    t0 = time.time()
    proba = fit_predict(name, tr, te)
    pred = proba.argmax(axis=1).astype(np.int8)
    oof[te], oof_p[te] = pred, proba
    r = evaluate(y[te], pred)
    r.update({"fold": k, "n_test": len(te), "seconds": time.time() - t0})
    rows.append(r)
    print(f"fold {k}: macro F1 {r['macro_f1']:.4f} | accuracy {r['accuracy']:.4f} | "
          f"binary F1 {r['binary_f1']:.4f} | Normal recall {r['normal_recall']:.4f} ({r['seconds']:.0f} s)", flush=True)

res = pd.DataFrame(rows)
res.to_csv(f"reports/cv/{name}_folds.csv", index=False)
np.save(f"reports/cv/{name}_oof.npy", oof)
np.save(f"reports/cv/{name}_oof_proba.npy", oof_p)

show = ["accuracy", "macro_f1", "weighted_f1", "binary_f1", "normal_recall",
        "recall_DoS", "recall_Analysis", "recall_Backdoor", "recall_Fuzzers"]
primary = res[res["fold"] >= 1]                 # folds 1-4: never used for model selection
print(f"\n{name}: PRIMARY (folds 1 to 4, not used for selection), mean +/- std over folds")
print(pd.DataFrame({"mean": primary[show].mean(), "std": primary[show].std()}).round(4).to_string())
print(f"fold 0 (used for selection, reference): macro F1 {res.loc[0, 'macro_f1']:.4f}")