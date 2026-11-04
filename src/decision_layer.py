import numpy as np
import pandas as pd
from engine import Engine
from features import CLASSES

D = "data/processed/"
eng = Engine()


def get_scores(X):
    proba = eng.clf.predict_proba(X)
    dec = eng.decide(X)
    return {"severity": dec["severity_score"].to_numpy(float),
            "p_attack": 1.0 - proba[:, 0]}


Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
Xt = pd.read_parquet(D + "X_test_eval.parquet")
yt = pd.read_parquet(D + "y_test_eval.parquet")["y"].to_numpy()
gt = pd.read_parquet(D + "test_eval.parquet")["group"].astype("uint64")

sv, st = get_scores(Xv), get_scores(Xt)
dev = (gt % 2 == 0).to_numpy()
fin = ~dev
print("validation rows:", len(yv), "| test dev rows:", int(dev.sum()), "| test final rows:", int(fin.sum()))


def threshold_for(s, y, budget):
    """Alert if score > t, with t chosen so about `budget` per 1000 Normal events alert."""
    return float(np.quantile(s[y == 0], 1 - budget / 1000.0))


def rates(s, y, t):
    alert = s > t
    att = y != 0
    return 1000 * alert[~att].mean(), alert[att].mean()


def boot(s, y, t, n=500, seed=0):
    rng = np.random.default_rng(seed)
    att = y != 0
    ia, inn = np.where(att)[0], np.where(~att)[0]
    fa, rc = [], []
    for _ in range(n):
        a = rng.choice(ia, len(ia)); b = rng.choice(inn, len(inn))
        fa.append(1000 * (s[b] > t).mean()); rc.append((s[a] > t).mean())
    return np.percentile(fa, [2.5, 97.5]), np.percentile(rc, [2.5, 97.5])


PI = 0.05   # assumed share of attacks in real traffic, for the precision column
rows = []
for rank in ["severity", "p_attack"]:
    s_dev, y_dev = st[rank][dev], yt[dev]
    s_fin, y_fin = st[rank][fin], yt[fin]
    if rank == "severity":
        fa, rc = rates(s_fin, y_fin, 64.95)
        rows.append({"ranking": rank, "budget": "current rule (score>=65)", "threshold": 65.0,
                     "dev_FA": rates(s_dev, y_dev, 64.95)[0], "dev_recall": rates(s_dev, y_dev, 64.95)[1],
                     "final_FA": fa, "final_recall": rc, "val_chosen_final_FA": np.nan,
                     "precision_at_5pct_attacks": rc * PI / (rc * PI + fa / 1000 * (1 - PI))})
    for budget in [10, 20, 50, 100]:
        t = threshold_for(s_dev, y_dev, budget)
        t_val = threshold_for(sv[rank], yv, budget)
        d_fa, d_rc = rates(s_dev, y_dev, t)
        f_fa, f_rc = rates(s_fin, y_fin, t)
        v_fa, _ = rates(s_fin, y_fin, t_val)
        (fl, fh), (rl, rh) = boot(s_fin, y_fin, t)
        rows.append({"ranking": rank, "budget": budget, "threshold": round(t, 3),
                     "dev_FA": d_fa, "dev_recall": d_rc, "final_FA": f_fa,
                     "final_FA_CI": f"{fl:.0f}-{fh:.0f}", "final_recall": f_rc,
                     "final_recall_CI": f"{rl:.3f}-{rh:.3f}", "val_chosen_final_FA": v_fa,
                     "precision_at_5pct_attacks": f_rc * PI / (f_rc * PI + f_fa / 1000 * (1 - PI))})

res = pd.DataFrame(rows)
res.to_csv("reports/decision_layer.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print()
print(res.round(3).to_string(index=False))

# which attack classes still alert at the 20-per-1000 budget (severity ranking, final half)
t20 = threshold_for(st["severity"][dev], yt[dev], 20)
alert = st["severity"][fin] > t20
true = pd.Series([CLASSES[k] for k in yt[fin]])
print(f"\nSHARE OF EACH TRUE CLASS THAT ALERTS at the 20 per 1000 budget (threshold {t20:.1f})")
print(pd.Series(alert).groupby(true).mean().round(3).reindex(CLASSES).to_string())