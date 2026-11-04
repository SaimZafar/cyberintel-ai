import os
import sys
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/cleaned.csv"
print("file:", PATH, "| size:", round(os.path.getsize(PATH) / 1e6, 1), "MB")

head = pd.read_csv(PATH, nrows=5)
cols = head.columns.tolist()
official = pd.read_csv("data/raw/UNSW_NB15_training-set.csv", nrows=0).columns.tolist()
print("\ncolumns (%d): %s" % (len(cols), cols))
print("same columns as the official training file:", cols == official)
print("missing vs official:", [c for c in official if c not in cols])
print("extra vs official:  ", [c for c in cols if c not in official])
print("\nfirst 2 rows:")
print(head.head(2).T.to_string())

skip = {"id", "attack_cat", "label"}
feat = [c for c in cols if c not in skip]
n, nulls = 0, None
lab, cat, hashes, ids = None, None, [], []
for chunk in pd.read_csv(PATH, chunksize=200_000, low_memory=False):
    n += len(chunk)
    nl = chunk.isna().sum()
    nulls = nl if nulls is None else nulls.add(nl, fill_value=0)
    if "label" in chunk:
        v = chunk["label"].value_counts()
        lab = v if lab is None else lab.add(v, fill_value=0)
    if "attack_cat" in chunk:
        v = chunk["attack_cat"].astype(str).value_counts()
        cat = v if cat is None else cat.add(v, fill_value=0)
    if "id" in chunk:
        ids.append(chunk["id"].to_numpy())
    hashes.append(pd.util.hash_pandas_object(chunk[feat], index=False).to_numpy())

import numpy as np
h = np.concatenate(hashes)
print("\nrows:", n)
if lab is not None:
    print("label counts:", lab.astype(int).to_dict())
if cat is not None:
    print("attack_cat counts:\n", cat.astype(int).sort_values(ascending=False).to_string())
nn = nulls[nulls > 0]
print("\ncolumns with missing values:", nn.astype(int).to_dict() if len(nn) else "none")
print("duplicate feature rows (approximate): %d (%.1f%%)" % (n - len(np.unique(h)), 100 * (1 - len(np.unique(h)) / n)))
if ids:
    allid = np.concatenate(ids)
    print("id column unique:", len(np.unique(allid)) == len(allid), "| min/max:", allid.min(), allid.max())