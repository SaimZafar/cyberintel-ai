import pandas as pd

train = pd.read_parquet("data/raw/UNSW_NB15_training-set.parquet")
test = pd.read_parquet("data/raw/UNSW_NB15_testing-set.parquet")

print("train:", train.shape, "test:", test.shape)
print(train.columns.tolist())

needed = ["sttl", "dttl", "ct_state_ttl", "ct_srv_src", "ct_dst_ltm",
          "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"]
print("\nmissing important columns:", [c for c in needed if c not in train.columns])

print("\ntrain classes:\n", train["attack_cat"].value_counts())
print("\ntest classes:\n", test["attack_cat"].value_counts())
print("\nmissing values:", int(train.isna().sum().sum()), int(test.isna().sum().sum()))
print("\ncategorical:", train[["proto", "service", "state"]].nunique().to_dict())
print("\nduplicate rows in train:", int(train.drop(columns=["id"], errors="ignore").duplicated().sum()))