import numpy as np
import pandas as pd

CLASSES = ["Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
           "Analysis", "Backdoor", "Shellcode", "Worms"]

CAT_COLS = ["proto", "service", "state"]

# columns with huge values and long tails: we take log(1 + x)
LOG_COLS = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
            "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "stcpb", "dtcpb",
            "tcprtt", "synack", "ackdat", "smean", "dmean", "response_body_len"]

# columns that are not model inputs
NON_FEATURES = ["id", "attack_cat", "label", "group", "seen_in_train"]

def learn_categories(df, min_count=300):
    """For proto, service, state: keep only values that appear often enough."""
    cats = {}
    for c in CAT_COLS:
        counts = df[c].astype(str).value_counts()
        cats[c] = counts[counts >= min_count].index.tolist()
    return cats


def make_features(df, cats):
    """Turn raw rows into a numeric table the models can use."""
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns]).copy()

    # 1) log transform for the skewed number columns
    for c in LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))

    # 2) text columns -> 0/1 columns; rare or unknown values become "other"
    for c in CAT_COLS:
        values = X[c].astype(str)
        values = values.where(values.isin(cats[c]), "other")
        for v in cats[c] + ["other"]:
            X[f"{c}_{v}"] = (values == v).astype("float32")
        X = X.drop(columns=[c])

    return X.astype("float32")


def get_labels(df):
    """Returns (class number 0-9, binary label 0/1)."""
    ids = {c: i for i, c in enumerate(CLASSES)}
    return df["attack_cat"].map(ids).to_numpy(), df["label"].to_numpy()