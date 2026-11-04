import json
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
MAX_EPOCHS, PATIENCE = 40, 6

Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
scaler = StandardScaler().fit(Xf)                       # training part only
Af = torch.tensor(scaler.transform(Xf), dtype=torch.float32)
Av = torch.tensor(scaler.transform(Xv), dtype=torch.float32)
Tf, Tv = torch.tensor(yf, dtype=torch.long), torch.tensor(yv, dtype=torch.long)
counts = np.bincount(yf, minlength=K)

rng = np.random.RandomState(7)
configs = [dict(hidden=[256, 128], dropout=0.2, lr=1e-3, wd=1e-4, batch=512, power=0.5)]   # config 0 = earlier MLP
for _ in range(10):
    configs.append(dict(
        hidden=[[128, 64], [256, 128], [512, 256], [256, 128, 64], [512, 256, 128]][rng.randint(5)],
        dropout=float(rng.choice([0.1, 0.2, 0.3, 0.4])),
        lr=float(rng.choice([3e-4, 1e-3, 3e-3])),
        wd=float(rng.choice([0.0, 1e-4, 1e-3])),
        batch=int(rng.choice([256, 512, 1024])),
        power=float(rng.choice([0.25, 0.5, 0.75]))))


def build(cfg):
    layers, d = [], Af.shape[1]
    for h in cfg["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg["dropout"])]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, K))


def train(cfg, seed, save=None):
    torch.manual_seed(seed)
    model = build(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    w = torch.tensor((counts.max() / counts) ** cfg["power"], dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best, state, bad, n = -1.0, None, 0, len(Tf)
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss_fn(model(Af[idx]), Tf[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Av).argmax(1).numpy()
        f1 = f1_score(yv, pred, average="macro")
        if f1 > best:
            best, bad, state, best_pred = f1, 0, {k: v.clone() for k, v in model.state_dict().items()}, pred
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if save:
        torch.save(state, save)
    return best, best_pred, time.time() - t0


rows = []
for i, cfg in enumerate(configs):
    f1, pred, sec = train(cfg, seed=0)
    att = yv != 0
    rows.append({"id": i, **{k: str(v) if k == "hidden" else v for k, v in cfg.items()},
                 "macro_f1": f1, "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
                 "normal_recall": float((pred[~att] == 0).mean()), "seconds": sec})
    print(f"config {i}: val macro F1 {f1:.4f} ({sec:.0f} s)", flush=True)

res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
res.to_csv("reports/mlp_tuning.csv", index=False)
pd.set_option("display.width", 250)
print()
print(res.round(4).head(6).to_string(index=False))

best_cfg = configs[int(res.iloc[0]["id"])]
print("\nrerunning the best config with 3 seeds:", best_cfg)
final = []
for s in range(3):
    f1, pred, sec = train(best_cfg, seed=s, save=f"models/mlp_best_seed{s}.pt")
    att = yv != 0
    final.append({"seed": s, "macro_f1": f1, "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
                  "normal_recall": float((pred[~att] == 0).mean())})
    print(f"  seed {s}: {f1:.4f}", flush=True)
fin = pd.DataFrame(final)
print(f"\nbest MLP config, 3 seeds: macro F1 {fin['macro_f1'].mean():.4f} +/- {fin['macro_f1'].std():.4f} "
      f"| binary F1 {fin['binary_f1'].mean():.4f} | Normal recall {fin['normal_recall'].mean():.4f}")
print("XGBoost v1 (validation):    macro F1 0.6567 | binary F1 0.9666 | Normal recall 0.8838")
json.dump({"config": best_cfg, "val_macro_f1_mean": float(fin["macro_f1"].mean())},
          open("models/mlp_best.json", "w"), indent=2)