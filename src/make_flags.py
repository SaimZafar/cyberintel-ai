from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW = "data/raw/"
OUT = Path("reports/release")
OUT.mkdir(parents=True, exist_ok=True)

tr = pd.read_csv(RAW + "UNSW_NB15_training-set.csv")
te = pd.read_csv(RAW + "UNSW_NB15_testing-set.csv")
feats = [c for c in tr.columns if c not in ("id", "attack_cat", "label")]
assert feats == [c for c in te.columns if c not in ("id", "attack_cat", "label")]

# identical feature rows share one group id (same hashing as make_splits.py)
for d in (tr, te):
    d["group"] = pd.util.hash_pandas_object(d[feats], index=False).astype("uint64")
tr["partition"], te["partition"] = "train", "test"


def majority(d):
    c = (d.groupby(["group", "attack_cat"]).size().reset_index(name="n")
         .sort_values("n", ascending=False).drop_duplicates("group").set_index("group"))
    return c["attack_cat"], d.groupby("group").size()


maj = {"train": majority(tr), "test": majority(te)}
for d, other in ((tr, "test"), (te, "train")):
    g = d.groupby("group")
    d["n_copies_in_partition"] = g["group"].transform("size")
    d["n_labels_in_partition"] = g["attack_cat"].transform("nunique")
    d["is_repeat_of_earlier_row"] = d.duplicated(subset="group", keep="first")
    o_label, o_size = maj[other]
    d["n_copies_in_other_partition"] = d["group"].map(o_size).fillna(0).astype(int)
    d["other_partition_majority_label"] = d["group"].map(o_label)

df = pd.concat([tr, te], ignore_index=True)
df["n_labels_overall"] = df.groupby("group")["attack_cat"].transform("nunique")
df["conflicting_labels_overall"] = df["n_labels_overall"] > 1
df["conflicting_labels_in_partition"] = df["n_labels_in_partition"] > 1
df["has_copy_in_other_partition"] = df["n_copies_in_other_partition"] > 0
df["label_matches_other_majority"] = (df["attack_cat"] == df["other_partition_majority_label"]).where(
    df["has_copy_in_other_partition"])

# group-aware 5-fold assignment for the training partition
t = df["partition"] == "train"
codes = pd.factorize(df.loc[t, "group"])[0]
fold = pd.Series(-1, index=df.index)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for k, (_, va) in enumerate(sgkf.split(df.loc[t], df.loc[t, "attack_cat"], groups=codes)):
    fold.loc[df.index[t][va]] = k
df["train_fold_group_aware"] = fold

cols = ["partition", "id", "attack_cat", "n_copies_in_partition", "is_repeat_of_earlier_row",
        "n_labels_in_partition", "conflicting_labels_in_partition", "n_labels_overall",
        "conflicting_labels_overall", "has_copy_in_other_partition",
        "n_copies_in_other_partition", "other_partition_majority_label",
        "label_matches_other_majority", "train_fold_group_aware"]
df["group"] = df["group"].astype(str)
df[["group"] + cols].to_csv(OUT / "unsw_nb15_row_flags.csv", index=False)

pct = lambda s: round(100 * float(s.mean()), 1)
print("rows:", len(df))
print(df.groupby("partition").agg(
    rows=("id", "size"),
    repeat_rows_pct=("is_repeat_of_earlier_row", pct),
    conflict_in_partition_pct=("conflicting_labels_in_partition", pct),
    conflict_overall_pct=("conflicting_labels_overall", pct),
    copy_in_other_pct=("has_copy_in_other_partition", pct)).to_string())
print("\nPER CLASS (percent of that class's rows)")
print(df.groupby("attack_cat").agg(
    rows=("id", "size"),
    repeat=("is_repeat_of_earlier_row", pct),
    conflict_overall=("conflicting_labels_overall", pct),
    copy_in_other=("has_copy_in_other_partition", pct)).to_string())
m = df[df["has_copy_in_other_partition"]]
print("\nrows with a copy in the other partition:", len(m),
      "| label matches the other partition's majority:", pct(m["label_matches_other_majority"].astype(float)), "%")
print("saved", OUT / "unsw_nb15_row_flags.csv")