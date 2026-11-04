import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW, OUT = "data/raw/", "data/processed/"
train = pd.read_csv(RAW + "UNSW_NB15_training-set.csv")
test = pd.read_csv(RAW + "UNSW_NB15_testing-set.csv")

# id is different for every row, so it must NOT be used to find identical rows
feats = [c for c in train.columns if c not in ("id", "attack_cat", "label")]

# identical feature rows share one group id
train["group"] = pd.util.hash_pandas_object(train[feats], index=False).astype("uint64")
test["group"] = pd.util.hash_pandas_object(test[feats], index=False).astype("uint64")

# 1) train / validation split: identical rows stay together
codes = pd.factorize(train["group"])[0]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fit_idx, val_idx = next(sgkf.split(train, train["attack_cat"], groups=codes))
fit, val = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()

# 2) demo pool: unique test rows never seen in training
train_groups = set(train["group"])
unseen = ~test["group"].isin(train_groups)
unique = ~test.duplicated(subset="group", keep=False)
eligible = test[unseen & unique]
parts = []
for cls, g in eligible.groupby("attack_cat", observed=True):
    n = min(100, max(1, len(g) // 2))
    parts.append(g.sample(n, random_state=42))
demo = pd.concat(parts)

# 3) test set for reporting (demo events removed)
test_eval = test[~test["group"].isin(demo["group"])].copy()
test_eval["seen_in_train"] = test_eval["group"].isin(train_groups)

# safety checks
assert not (set(fit["group"]) & set(val["group"])), "leak between fit and val"
assert not (set(demo["group"]) & train_groups), "demo overlaps training"
assert fit["group"].nunique() < len(fit), "groups look wrong: no duplicates found"

fit.to_parquet(OUT + "fit.parquet")
val.to_parquet(OUT + "val.parquet")
test_eval.to_parquet(OUT + "test_eval.parquet")
demo.to_parquet(OUT + "demo_pool.parquet")

print("fit:", fit.shape, "val:", val.shape)
print("test_eval:", test_eval.shape, "| unseen in train:", int((~test_eval["seen_in_train"]).sum()))
print("demo pool:", demo.shape)
print("\nfit classes:\n", fit["attack_cat"].value_counts().to_dict())
print("val classes:\n", val["attack_cat"].value_counts().to_dict())
print("demo classes:\n", demo["attack_cat"].value_counts().to_dict())