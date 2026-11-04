import pandas as pd
from engine import Engine
from features import CLASSES

D = "data/processed/"
Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
yb = pd.read_parquet(D + "y_val.parquet")["y_bin"].to_numpy()
true = pd.Series([CLASSES[k] for k in yv])
att = pd.Series(yb == 1)
normal = ~att

variants = {
    "current (0.4/0.3/0.3, thr 0.95)": ((0.4, 0.3, 0.3), 0.95),
    "no anomaly in severity (0.5/0/0.5)": ((0.5, 0.0, 0.5), 0.95),
    "less anomaly (0.5/0.2/0.3)": ((0.5, 0.2, 0.3), 0.95),
    "anomaly incidents off": ((0.4, 0.3, 0.3), 1.01),
    "stricter anomaly (thr 0.99)": ((0.4, 0.3, 0.3), 0.99),
}

eng = Engine()
rows = []
for name, (w, thr) in variants.items():
    out = eng.decide(Xv, weights=w, threshold=thr)
    alert = out["severity_level"].isin(["High", "Critical"])
    tp = int((alert & att).sum())
    crit = out["severity_level"].eq("Critical")
    rows.append({
        "variant": name,
        "alert_precision": tp / max(int(alert.sum()), 1),
        "alert_recall": tp / int(att.sum()),
        "false_alerts_per_1000": 1000 * int((alert & normal).sum()) / int(normal.sum()),
        "anomaly_false_incidents_%": 100 * int((out["anomaly_only"] & normal).sum()) / int(normal.sum()),
        "attacks_caught_by_anomaly": int((out["anomaly_only"] & att).sum()),
        "Generic_critical": crit[true == "Generic"].mean(),
        "Worms_critical": crit[true == "Worms"].mean(),
        "mean_score_Generic": out.loc[true == "Generic", "severity_score"].mean(),
        "mean_score_Worms": out.loc[true == "Worms", "severity_score"].mean(),
    })

result = pd.DataFrame(rows)
result.to_csv("reports/severity_ablation.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
print(result.round(3).to_string(index=False))