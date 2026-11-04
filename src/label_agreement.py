import pandas as pd

D = "data/processed/"
tr = pd.concat([pd.read_parquet(D + "fit.parquet")[["group", "attack_cat"]],
                pd.read_parquet(D + "val.parquet")[["group", "attack_cat"]]])
te = pd.read_parquet(D + "test_eval.parquet")[["group", "attack_cat"]]

majority = tr.groupby("group")["attack_cat"].agg(lambda s: s.value_counts().index[0])
n_labels = tr.groupby("group")["attack_cat"].nunique()

both = te[te["group"].isin(majority.index)].copy()
both["train_label"] = both["group"].map(majority)
both["train_group_has_conflict"] = both["group"].map(n_labels) > 1
both["agree"] = both["attack_cat"] == both["train_label"]

print("test rows with an identical pattern in training:", len(both), "of", len(te))
print("label agrees with the training majority label:   ", round(both["agree"].mean(), 3))
print("  if that training pattern had ONE label:        ",
      round(both.loc[~both["train_group_has_conflict"], "agree"].mean(), 3),
      "(n =", int((~both["train_group_has_conflict"]).sum()), ")")
print("  if that training pattern had CONFLICTING labels:",
      round(both.loc[both["train_group_has_conflict"], "agree"].mean(), 3),
      "(n =", int(both["train_group_has_conflict"].sum()), ")")

print("\nBY TEST CLASS")
by = both.groupby("attack_cat")["agree"].agg(rows="size", agree="mean").round(3)
print(by.to_string())

print("\nMOST COMMON DISAGREEMENTS (test label -> training majority label)")
bad = both[~both["agree"]]
print(bad.groupby(["attack_cat", "train_label"]).size().sort_values(ascending=False).head(10).to_string())