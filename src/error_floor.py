import pandas as pd

df = pd.read_csv("reports/release/unsw_nb15_row_flags.csv", dtype={"group": str},
                 usecols=["group", "partition", "attack_cat"])


def ceilings(d):
    c = (d.groupby(["group", "attack_cat"]).size().rename("n").reset_index()
         .sort_values(["group", "n", "attack_cat"], ascending=[True, False, True]))
    top = c.drop_duplicates("group")                 # most common label of each pattern
    ties = int((c.groupby("group")["n"].apply(lambda s: (s == s.max()).sum() > 1)).sum())
    acc = top["n"].sum() / len(d)
    rec = (top.groupby("attack_cat")["n"].sum() / d.groupby("attack_cat").size()).round(3)
    return acc, rec, ties


out = {}
for name, d in [("pooled", df), ("train partition", df[df["partition"] == "train"]),
                ("test partition", df[df["partition"] == "test"])]:
    acc, rec, ties = ceilings(d)
    out[name] = rec
    print(f"{name:<16} accuracy ceiling {acc:.4f} | patterns with tied labels: {ties}")
print()
print("per-class recall ceiling")
print(pd.DataFrame(out).to_string())
print("\nBinary view (attack vs normal) ceiling")
b = df.assign(y=(df["attack_cat"] != "Normal").map({True: "attack", False: "normal"}))
for name, d in [("pooled", b), ("train partition", b[b["partition"] == "train"]),
                ("test partition", b[b["partition"] == "test"])]:
    c = d.groupby(["group", "y"]).size().groupby("group").max().sum() / len(d)
    print(f"{name:<16} binary accuracy ceiling {c:.4f}")