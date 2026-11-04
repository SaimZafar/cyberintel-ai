from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from features import CLASSES

D = "data/processed/"
K = len(CLASSES)
Xf = pd.read_parquet(D + "X_fit.parquet")
Xv = pd.read_parquet(D + "X_val.parquet")
fit_raw = pd.read_parquet(D + "fit.parquet")
val_raw = pd.read_parquet(D + "val.parquet")
X = pd.concat([Xf, Xv], ignore_index=True)
y = np.r_[pd.read_parquet(D + "y_fit.parquet")["y"].to_numpy(),
          pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()]
ids = np.r_[fit_raw["id"].to_numpy(), val_raw["id"].to_numpy()]
flags = pd.read_csv("reports/release/unsw_nb15_row_flags.csv", dtype={"group": str},
                    usecols=["partition", "id", "train_fold_group_aware"])
flags = flags[flags["partition"] == "train"].set_index("id")["train_fold_group_aware"]
fold = flags.loc[ids].to_numpy()

p_att = 1.0 - np.load("reports/cv/xgb_oof_proba.npy").astype(np.float64)[:, 0]

# ---- out-of-fold anomaly scores: forest trained on Normal rows of the other folds only ----
path = Path("reports/cv/iforest_oof_anom.npy")
if path.exists():
    anom = np.load(path)
else:
    anom = np.zeros(len(y))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        norm = tr[y[tr] == 0].copy()
        np.random.RandomState(42).shuffle(norm)
        cut = int(0.7 * len(norm))
        forest = IsolationForest(n_estimators=300, random_state=42, n_jobs=6).fit(X.iloc[norm[:cut]])
        ref = np.sort(forest.score_samples(X.iloc[norm[cut:]]))
        s = forest.score_samples(X.iloc[te])
        anom[te] = (len(ref) - np.searchsorted(ref, s, side="right")) / len(ref)
        print("anomaly scores done for fold", f, flush=True)
    np.save(path, anom)

normal, attack = y == 0, y != 0


def crossfit(score, budget):
    """Threshold for each fold comes from the Normal rows of the OTHER folds."""
    alert = np.zeros(len(y), dtype=bool)
    for f in range(5):
        t = np.quantile(score[(fold != f) & normal], 1 - budget / 1000.0)
        alert[fold == f] = score[fold == f] > t
    return alert


rules = {
    "classifier only, budget 20": crossfit(p_att, 20),
    "classifier 15 + anomaly 5": crossfit(p_att, 15) | crossfit(anom, 5),
    "classifier 10 + anomaly 10": crossfit(p_att, 10) | crossfit(anom, 10),
    "anomaly only, budget 20": crossfit(anom, 20),
}
base = rules["classifier only, budget 20"]
hard = ["DoS", "Analysis", "Backdoor", "Fuzzers", "Exploits", "Generic"]
rows = []
for name, a in rules.items():
    fa = [1000 * a[(fold == f) & normal].mean() for f in range(5)]
    rc = [a[(fold == f) & attack].mean() for f in range(5)]
    r = {"rule": name, "FA_per_1000": 1000 * a[normal].mean(), "FA_fold_min": min(fa), "FA_fold_max": max(fa),
         "attack_recall": a[attack].mean(), "recall_fold_min": min(rc), "recall_fold_max": max(rc),
         "extra_attacks_vs_base": int((a & ~base & attack).sum()), "attacks_lost_vs_base": int((~a & base & attack).sum())}
    for c in hard:
        r[c] = a[y == CLASSES.index(c)].mean()
    rows.append(r)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 30)
print("\nOUT-OF-FOLD ALERT RULES (all 5 folds; thresholds cross-fitted, test partition never used)")
print(pd.DataFrame(rows).round(3).to_string(index=False))

t_all = float(np.quantile(p_att[normal], 0.98))
print(f"\npooled out-of-fold threshold for a budget of 20 per 1000: {t_all:.3f} "
      f"(the earlier test-chosen value was 0.956)")