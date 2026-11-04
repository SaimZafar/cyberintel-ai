import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

D = "data/processed/"
tr = pd.concat([pd.read_parquet(D + "fit.parquet")[["group", "attack_cat"]],
                pd.read_parquet(D + "val.parquet")[["group", "attack_cat"]]], ignore_index=True)
te = pd.read_parquet(D + "test_eval.parquet")[["group", "attack_cat"]]


def majority(ref):
    c = ref.groupby(["group", "attack_cat"]).size().reset_index(name="n")
    total = c.groupby("group")["n"].sum()
    top = c.sort_values("n", ascending=False).drop_duplicates("group").set_index("group")
    return top["attack_cat"], total


def agreement(ref, probe, min_copies):
    maj, total = majority(ref)
    p = probe[probe["group"].isin(maj.index)].copy()
    p["maj"] = p["group"].map(maj)
    p["copies"] = p["group"].map(total)
    p = p[p["copies"] >= min_copies]
    p["agree"] = p["maj"] == p["attack_cat"]
    return p.groupby("attack_cat")["agree"].agg(n="size", agree="mean")


for min_copies in [1, 3]:
    cross = agreement(tr, te, min_copies)
    within = []
    for seed in range(5):
        a, b = train_test_split(tr, test_size=0.2, stratify=tr["attack_cat"], random_state=seed)
        within.append(agreement(a, b, min_copies))
    w_n = pd.concat([w["n"] for w in within], axis=1).mean(axis=1)
    w_ag = pd.concat([w["agree"] for w in within], axis=1).mean(axis=1)

    t = pd.DataFrame({"cross_n": cross["n"], "cross_agree": cross["agree"],
                      "within_n": w_n, "within_agree": w_ag}).dropna()
    t["gap"] = t["within_agree"] - t["cross_agree"]
    ok = t[(t["cross_n"] >= 30) & (t["within_n"] >= 30)]
    print(f"\n=== patterns with at least {min_copies} training copies ===")
    print(t.round(3).to_string())
    wts = ok["cross_n"] / ok["cross_n"].sum()
    print(f"\nclasses with 30+ rows in both: {list(ok.index)}")
    print("cross-partition agreement (weighted by its own class mix): ",
          round(float((ok['cross_agree'] * wts).sum()), 3))
    print("within-partition agreement re-weighted to that class mix:   ",
          round(float((ok['within_agree'] * wts).sum()), 3))