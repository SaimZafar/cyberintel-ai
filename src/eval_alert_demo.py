import pandas as pd
from engine import Engine
from features import CLASSES

D = "data/processed/"
X = pd.read_parquet(D + "X_demo_pool.parquet")
y = pd.read_parquet(D + "y_demo_pool.parquet")["y"].to_numpy()

out = Engine().decide(X)
out["true"] = [CLASSES[k] for k in y]

print("share of each true class that raises an alert:")
print(out.groupby("true")["alert"].mean().reindex(CLASSES).round(3).to_string())

att = out["true"] != "Normal"
print("\nattacks that alert:      ", round(float(out.loc[att, "alert"].mean()), 3), "(old rule: 0.729)")
print("Normal events that alert:", round(float(out.loc[~att, "alert"].mean()), 3), "(old rule: 0.14)")