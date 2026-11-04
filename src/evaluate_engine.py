import pandas as pd
from engine import Engine
from features import CLASSES

D = "data/processed/"
Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
yb = pd.read_parquet(D + "y_val.parquet")["y_bin"].to_numpy()

out = Engine().decide(Xv)
out["true"] = [CLASSES[k] for k in yv]
att = pd.Series(yb == 1, index=out.index)
normal = ~att

print("INCIDENTS (events saved for the analyst)")
print("  attacks that create an incident:      ", round(out.loc[att, "incident"].mean(), 4))
print("  normal events that create an incident:", round(out.loc[normal, "incident"].mean(), 4))

missed = att & (out["category"] == "Normal")
caught = missed & out["anomaly_only"]
extra = normal & out["anomaly_only"]
print("\nWHAT THE ANOMALY DETECTOR ADDS")
print(f"  attacks the classifier called Normal: {int(missed.sum())}")
print(f"  of those, caught by the anomaly score: {int(caught.sum())} ({caught.sum() / max(missed.sum(), 1):.1%})")
print(f"  extra false incidents it causes:       {int(extra.sum())} ({extra.sum() / normal.sum():.1%} of normal events)")

alert = out["severity_level"].isin(["High", "Critical"])
tp = int((alert & att).sum())
print("\nALERTS (High or Critical, these notify the phone)")
print("  alert precision:", round(tp / max(int(alert.sum()), 1), 4))
print("  alert recall:   ", round(tp / int(att.sum()), 4))
print("  false alerts per 1000 normal events:", round(1000 * int((alert & normal).sum()) / int(normal.sum()), 1))

table = pd.crosstab(out["true"], out["severity_level"], normalize="index")
table = table.reindex(index=CLASSES, columns=["Low", "Medium", "High", "Critical"], fill_value=0)
print("\nSEVERITY LEVEL BY TRUE CLASS (each row adds up to 1)")
print(table.round(3).to_string())
table.round(4).to_csv("reports/severity_by_class.csv")