import numpy as np
import pandas as pd
from engine import Engine
from features import CLASSES

D = "data/processed/"
X = pd.read_parquet(D + "X_demo_pool.parquet")
y = pd.read_parquet(D + "y_demo_pool.parquet")["y"].to_numpy()

out = Engine().decide(X)
out["true"] = [CLASSES[k] for k in y]
out["alert"] = out["severity_level"].isin(["High", "Critical"])

rows = []
for c in CLASSES:
    m = out["true"] == c
    wrong = out.loc[m & (out["category"] != c), "category"]
    rows.append({
        "class": c,
        "events": int(m.sum()),
        "correct": round(float((out.loc[m, "category"] == c).mean()), 3),
        "most_common_mistake": wrong.value_counts().index[0] if len(wrong) else "-",
        "alert_rate": round(float(out.loc[m, "alert"].mean()), 3),
    })
table = pd.DataFrame(rows)
print(table.to_string(index=False))

attack = out["true"] != "Normal"
print("\nattacks that raise an alert (High or Critical):", round(float(out.loc[attack, "alert"].mean()), 3))
print("normal events that raise an alert:             ", round(float(out.loc[~attack, "alert"].mean()), 3))
print("attacks the engine labels Normal:               ", round(float((out.loc[attack, "category"] == "Normal").mean()), 3))
print("overall category accuracy:                      ", round(float((out["category"] == out["true"]).mean()), 3))
table.to_csv("reports/demo_pool_results.csv", index=False)