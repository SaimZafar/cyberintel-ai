import numpy as np
import pandas as pd
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
MODELS = ["xgb", "lgbm", "rf", "mlp", "tabnet"]

fit_raw = pd.read_parquet(D + "fit.parquet")
val_raw = pd.read_parquet(D + "val.parquet")
y = np.r_[pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy(),
          pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()]
ids = np.r_[fit_raw["id"].to_numpy(), val_raw["id"].to_numpy()]
groups = np.r_[fit_raw["group"].to_numpy(), val_raw["group"].to_numpy()]
flags = pd.read_csv("reports/release/unsw_nb15_row_flags.csv", dtype={"group": str},
                    usecols=["partition", "id", "train_fold_group_aware"])
flags = flags[flags["partition"] == "train"].set_index("id")["train_fold_group_aware"]
fold = flags.loc[ids].to_numpy()
preds = {k: np.load(f"reports/cv/{k}_oof.npy").astype(int) for k in MODELS}

rows, ceil_acc, class_conf = [], [], []
for f in range(1, 5):                               # folds 1 to 4
    te = np.where(fold == f)[0]
    df = pd.DataFrame({"g": groups[te], "y": y[te]})
    size = df.groupby("g")["y"].transform("size").to_numpy()
    nlab = df.groupby("g")["y"].transform("nunique").to_numpy()
    top = df.groupby(["g", "y"]).size().groupby("g").max()
    maj = df["g"].map(top).to_numpy()
    ptype = np.where(nlab > 1, "conflicting", np.where(size > 1, "repeated, one label", "seen once"))
    ceil_acc.append(df.groupby("g")["y"].apply(lambda s: s.value_counts().iloc[0]).sum() / len(te))
    class_conf.append(pd.DataFrame({"class": [CLASSES[i] for i in y[te]], "conf": nlab > 1}))
    for k in MODELS:
        p = preds[k][te]
        wrong = p != y[te]
        for t in ["seen once", "repeated, one label", "conflicting"]:
            m = ptype == t
            rows.append({"fold": f, "model": k, "type": t, "rows": int(m.sum()), "wrong": int(wrong[m].sum())})

r = pd.DataFrame(rows).groupby(["model", "type"])[["rows", "wrong"]].sum().reset_index()
r["accuracy"] = 1 - r["wrong"] / r["rows"]
tot = r.groupby("model")[["rows", "wrong"]].transform("sum")
r["share_of_rows"] = r["rows"] / tot["rows"]
r["share_of_errors"] = r["wrong"] / tot["wrong"]
pd.set_option("display.width", 250)
print("ERRORS BY PATTERN TYPE (folds 1 to 4 pooled; pattern = identical 42 features)")
print(r.pivot(index="model", columns="type", values=["share_of_rows", "accuracy", "share_of_errors"]).round(3).to_string())

ca = float(np.mean(ceil_acc))
print(f"\nACCURACY CEILING per fold (majority label of each pattern), mean over folds 1 to 4: {ca:.4f}")
acc = {k: float(np.mean([(preds[k][fold == f] == y[fold == f]).mean() for f in range(1, 5)])) for k in MODELS}
print(pd.DataFrame({"accuracy": acc, "gap_to_ceiling": {k: ca - v for k, v in acc.items()}}).round(4).to_string())

cc = pd.concat(class_conf).groupby("class")["conf"].mean().reindex(CLASSES)
rec = pd.DataFrame({k: [(preds[k][(fold >= 1) & (y == c)] == c).mean() for c in range(K)] for k in MODELS}, index=CLASSES)
out = rec.copy()
out.insert(0, "share_in_conflicting_patterns", cc)
print("\nPER CLASS: share of the class's rows in conflicting patterns, and recall")
print(out.round(3).to_string())
print("\nSpearman correlation across the 10 classes (conflict share vs recall):")
print({k: round(float(cc.corr(rec[k], method="spearman")), 3) for k in MODELS})