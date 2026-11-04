import pandas as pd

train = pd.read_csv("data/raw/UNSW_NB15_training-set.csv")
test = pd.read_csv("data/raw/UNSW_NB15_testing-set.csv")
print("train:", train.shape, "test:", test.shape)
print("train classes:", train["attack_cat"].value_counts().to_dict())
print("test classes: ", test["attack_cat"].value_counts().to_dict())
print("missing values:", int(train.isna().sum().sum()), int(test.isna().sum().sum()))
print("categorical:", train[["proto", "service", "state"]].nunique().to_dict())

# the id column is different for every row, so it must not be used to compare rows
feats = [c for c in train.columns if c not in ("id", "attack_cat", "label")]
print("number of feature columns:", len(feats))

dup = train.duplicated(subset=feats)
print(f"\nduplicate rows in train: {int(dup.sum())} ({dup.mean():.1%})   [34-feature version: 83,243 (47.5%)]")
print(pd.DataFrame({"total": train["attack_cat"].value_counts(),
                    "duplicates": train[dup]["attack_cat"].value_counts()}).fillna(0).astype(int))

train["h"] = pd.util.hash_pandas_object(train[feats], index=False)
test["h"] = pd.util.hash_pandas_object(test[feats], index=False)
c = train.groupby("h")["attack_cat"].nunique()
bad = c[c > 1].index
print(f"\nfeature patterns with conflicting labels: {len(bad)}, covering "
      f"{int(train['h'].isin(bad).sum())} rows   [34-feature: 1,397 patterns, 35,616 rows]")
print("test rows that also appear in train:", int(test["h"].isin(train["h"]).sum()),
      "of", len(test), "  [34-feature: 20,851]")
print("duplicate rows inside test:", int(test.duplicated(subset=feats).sum()), "  [34-feature: 34,101]")