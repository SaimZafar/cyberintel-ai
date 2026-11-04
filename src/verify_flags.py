import pandas as pd

RAW = "data/raw/"
tr = pd.read_csv(RAW + "UNSW_NB15_training-set.csv")
te = pd.read_csv(RAW + "UNSW_NB15_testing-set.csv")
tr["partition"], te["partition"] = "train", "test"
df = pd.concat([tr, te], ignore_index=True)
feats = [c for c in tr.columns if c not in ("id", "attack_cat", "label", "partition")]

fl = pd.read_csv("reports/release/unsw_nb15_row_flags.csv", dtype={"group": str})
m = df.merge(fl[["partition", "id", "group", "n_copies_in_partition", "conflicting_labels_overall",
                 "has_copy_in_other_partition", "n_copies_in_other_partition"]],
             on=["partition", "id"], how="left", validate="one_to_one")
print("rows:", len(m), "| rows without a flag:", int(m["group"].isna().sum()))

# 1) the number of distinct feature rows must equal the number of groups
n_distinct = len(m.drop_duplicates(subset=feats))
print("distinct feature rows:", n_distinct, "| groups:", m["group"].nunique(),
      "| match:", n_distinct == m["group"].nunique())

# 2) every group must be identical on all features
nun = m.groupby("group")[feats].nunique()
print("groups with any feature that differs inside the group:", int((nun > 1).any(axis=1).sum()))

# 3) the flags must agree with the raw data
cnt = m.groupby(["group", "partition"]).size().unstack(fill_value=0)
mm = m.join(cnt, on="group")
own = mm.apply(lambda r: r[r["partition"]], axis=1)
other = mm.apply(lambda r: r["test" if r["partition"] == "train" else "train"], axis=1)
print("n_copies_in_partition wrong:", int((own != mm["n_copies_in_partition"]).sum()))
print("n_copies_in_other_partition wrong:", int((other != mm["n_copies_in_other_partition"]).sum()))
nl = m.groupby("group")["attack_cat"].nunique()
print("conflict flag wrong:", int(((m["group"].map(nl) > 1) != m["conflicting_labels_overall"]).sum()))

# 4) look at three examples by eye
ex = m[m["conflicting_labels_overall"] & m["has_copy_in_other_partition"]]["group"].drop_duplicates().head(3)
for g in ex:
    print("\nexample group", g)
    print(m[m["group"] == g][["partition", "id", "attack_cat", "dur", "sbytes", "dbytes", "sttl", "ct_srv_src"]].to_string(index=False))