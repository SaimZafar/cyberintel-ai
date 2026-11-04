import numpy as np
import pandas as pd

RAW = "data/raw/"
CLEAN = RAW + "cleaned.csv"
CLASSES = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
           "Analysis", "Backdoor", "Shellcode", "Worms"]
CAT = {c.lower(): i for i, c in enumerate(CLASSES)}
CAT["backdoors"] = CAT["backdoor"]
CATCOLS = ["proto", "state", "service"]
RENAME = {"spkts": "Spkts", "dpkts": "Dpkts", "sload": "Sload", "sinpkt": "Sintpkt",
          "dinpkt": "Dintpkt", "sjit": "Sjit", "djit": "Djit", "smean": "smeansz",
          "dmean": "dmeansz", "response_body_len": "res_bdy_len"}


def frame(d, cols):
    out = pd.DataFrame(index=d.index)
    for c in sorted(cols):
        if c in CATCOLS:
            out[c] = d[c].astype(str).str.strip().str.lower()
        else:
            out[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    return out


def hashes(d, cols):
    return pd.util.hash_pandas_object(frame(d, cols), index=False).to_numpy()


def labels(s):
    return s.astype(str).str.strip().str.lower().map(CAT).fillna(-1).astype("int8").to_numpy()


head = pd.read_csv(CLEAN, nrows=1)
shared = [c for c in head.columns if c not in ("attack_cat", "Label")]
print("shared feature columns:", len(shared))

hs, ys = [], []
for ch in pd.read_csv(CLEAN, chunksize=250_000, low_memory=False):
    hs.append(hashes(ch, shared))
    ys.append(labels(ch["attack_cat"]))
h_full, y_full = np.concatenate(hs), np.concatenate(ys)
print("full capture rows:", len(y_full), "| unmapped labels:", int((y_full < 0).sum()),
      "| attack share: %.2f%%" % (100 * (y_full > 0).sum() / (y_full >= 0).sum()))

off = pd.concat([pd.read_csv(RAW + "UNSW_NB15_training-set.csv"),
                 pd.read_csv(RAW + "UNSW_NB15_testing-set.csv")], ignore_index=True)
y_off = labels(off["attack_cat"])
all42 = [c for c in off.columns if c not in ("id", "attack_cat", "label")]
h_off42 = hashes(off, all42)
off_r = off.rename(columns=RENAME)
assert set(shared) <= set(off_r.columns), set(shared) - set(off_r.columns)
h_off36 = hashes(off_r, shared)
print("official pooled rows:", len(y_off), "| attack share: %.2f%%" % (100 * (y_off > 0).mean()))


def stats(h, y, name):
    keep = y >= 0
    df = pd.DataFrame({"h": h[keep], "y": y[keep]})
    g = df.groupby("h")["y"]
    df["repeated"] = g.transform("size") > 1
    df["conflict"] = g.transform("nunique") > 1
    has_n = (df["y"] == 0).groupby(df["h"]).transform("any")
    has_a = (df["y"] != 0).groupby(df["h"]).transform("any")
    df["other_side"] = np.where(df["y"] == 0, has_a, has_n)
    r = df.groupby("y")[["repeated", "conflict", "other_side"]].mean().mul(100).round(1)
    r.insert(0, "rows", df.groupby("y").size())
    r.index = [CLASSES[i] for i in r.index]
    r.columns = [f"{name}: {c}" for c in r.columns]
    return r


t = pd.concat([stats(h_full, y_full, "FULL36"),
               stats(h_off36, y_off, "OFF36"),
               stats(h_off42, y_off, "OFF42")], axis=1).reindex(CLASSES)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 30)
print("\nrepeated = pattern appears more than once | conflict = pattern has 2+ classes")
print("other_side = for attacks: pattern also seen as Normal; for Normal: also seen as an attack\n")
print(t.to_string())
t.to_csv("reports/full_vs_official.csv")