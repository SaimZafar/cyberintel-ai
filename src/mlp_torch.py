import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
IDX = {c: i for i, c in enumerate(CLASSES)}
SEEDS = [0, 1, 2]
MAX_EPOCHS, PATIENCE, BATCH = 40, 6, 512

Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
yf = pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy()
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()

scaler = StandardScaler().fit(Xf)                 # training part only
Af = torch.tensor(scaler.transform(Xf), dtype=torch.float32)
Av = torch.tensor(scaler.transform(Xv), dtype=torch.float32)
Tf, Tv = torch.tensor(yf, dtype=torch.long), torch.tensor(yv, dtype=torch.long)
np.savez("models/mlp_scaler.npz", mean=scaler.mean_, scale=scaler.scale_, columns=np.array(Xf.columns))

counts = np.bincount(yf, minlength=K)
class_w = torch.tensor((counts.max() / counts) ** 0.5, dtype=torch.float32)   # same idea as XGBoost v1


def make_model():
    return nn.Sequential(nn.Linear(Af.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(128, K))


def metrics(y, pred):
    att = y != 0
    f1s = f1_score(y, pred, labels=list(range(K)), average=None, zero_division=0)
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "binary_f1": f1_score(att.astype(int), (pred != 0).astype(int)),
            "normal_recall": float((pred[~att] == 0).mean()),
            "f1_DoS": f1s[IDX["DoS"]], "f1_Analysis": f1s[IDX["Analysis"]],
            "f1_Backdoor": f1s[IDX["Backdoor"]]}


def median_ms(fn, n=300):
    for _ in range(20):
        fn()
    t = []
    for _ in range(n):
        a = time.perf_counter()
        fn()
        t.append((time.perf_counter() - a) * 1000)
    return float(np.median(t))


def run(seed):
    torch.manual_seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    best, best_state, bad, hist, n = -1.0, None, 0, [], len(Tf)
    start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm, total = torch.randperm(n), 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Af[idx]), Tf[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(Av)
            val_loss = loss_fn(logits, Tv).item()
            pred = logits.argmax(1).numpy()
        f1 = f1_score(yv, pred, average="macro")
        hist.append({"seed": seed, "epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_macro_f1": f1})
        if f1 > best:
            best, bad, best_epoch = f1, 0, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    seconds = time.time() - start
    model.load_state_dict(best_state)
    model.eval()
    path = f"models/mlp_seed{seed}.pt"
    torch.save(best_state, path)
    with torch.no_grad():
        pred = model(Av).argmax(1).numpy()
    r = metrics(yv, pred)
    r.update({"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch, "train_seconds": seconds,
              "size_mb": os.path.getsize(path) / 1e6, "pred": pred})
    print(f"seed {seed}: val macro F1 {r['macro_f1']:.4f} at epoch {best_epoch} ({seconds:.0f} s)", flush=True)
    return r, hist, model


results, curves, models = [], [], {}
for s in SEEDS:
    r, h, m = run(s)
    results.append(r)
    curves += h
    models[s] = m

# latency, one thread, one row (like the service)
torch.set_num_threads(1)
row = Av[:1]
with torch.inference_mode():
    mlp_ms = median_ms(lambda: models[SEEDS[0]](row))
xgb = joblib.load("models/classifier_v1.joblib")
xgb.set_params(n_jobs=1)
row_df = Xv.iloc[[0]]
xgb_ms = median_ms(lambda: xgb.predict_proba(row_df))
xgb_metrics = metrics(yv, xgb.predict(Xv))

res = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in results])
res.to_csv("reports/mlp_val_results.csv", index=False)
pd.DataFrame(curves).to_csv("reports/mlp_curves.csv", index=False)

cols = ["macro_f1", "weighted_f1", "binary_f1", "normal_recall", "f1_DoS", "f1_Analysis", "f1_Backdoor"]
table = pd.DataFrame({"MLP mean": res[cols].mean(), "MLP std": res[cols].std(),
                      "XGBoost v1": pd.Series(xgb_metrics)[cols]}).round(4)
pd.set_option("display.width", 200)
print("\nVALIDATION (test set not used)")
print(table.to_string())
n_params = sum(p.numel() for p in models[SEEDS[0]].parameters())
print(f"\nMLP: {n_params:,} parameters, {res['size_mb'].iloc[0]:.2f} MB, "
      f"train time {res['train_seconds'].mean():.0f} s per seed, best epochs {res['best_epoch'].tolist()}")
print(f"XGBoost v1: {os.path.getsize('models/classifier_v1.joblib') / 1e6:.1f} MB")
print(f"single-row latency, 1 thread: MLP {mlp_ms:.2f} ms | XGBoost predict_proba {xgb_ms:.2f} ms")

best_seed = int(res.loc[res["macro_f1"].idxmax(), "seed"])
rep = classification_report(yv, [r for r in results if r["seed"] == best_seed][0]["pred"],
                            labels=list(range(K)), target_names=CLASSES, zero_division=0, digits=3)
open("reports/mlp_val_report.txt", "w").write(rep)
print(f"\nper-class report for seed {best_seed} saved to reports/mlp_val_report.txt")

c = pd.DataFrame(curves)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for s, g in c.groupby("seed"):
    ax[0].plot(g["epoch"], g["train_loss"], label=f"train {s}")
    ax[0].plot(g["epoch"], g["val_loss"], "--", label=f"val {s}")
    ax[1].plot(g["epoch"], g["val_macro_f1"], label=f"seed {s}")
ax[0].set_title("loss (class-weighted)")
ax[1].set_title("validation macro F1")
ax[1].axhline(xgb_metrics["macro_f1"], color="k", linestyle=":", label="XGBoost v1")
ax[0].legend(fontsize=7)
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("reports/figures/mlp_curves.png", dpi=150)
print("saved reports/figures/mlp_curves.png")