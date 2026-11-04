import json
import urllib.request
import numpy as np
import pandas as pd

URL = "http://127.0.0.1:8000/detect"
demo = pd.read_parquet("data/processed/demo_pool.parquet")

rows = []
for _, row in demo.iterrows():
    event = json.loads(row.drop(["attack_cat", "label", "group"]).to_json())
    req = urllib.request.Request(URL, data=json.dumps(event).encode(),
                                 headers={"Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(req))
    rows.append({"true": row["attack_cat"], "pred": res["attack_category"],
                 "level": res["severity_level"], "ms": res["processing_ms"]})

df = pd.DataFrame(rows)
df["correct"] = df["true"] == df["pred"]
df["alert"] = df["level"].isin(["High", "Critical"])
df["detected"] = df["pred"] != "Normal"

summary = df.groupby("true").agg(events=("correct", "size"),
                                 exact_accuracy=("correct", "mean"),
                                 flagged_as_attack=("detected", "mean"),
                                 alert_rate=("alert", "mean")).round(3)
print(summary.to_string())
summary.to_csv("reports/demo_pool_results.csv")

att, normal = df["true"] != "Normal", df["true"] == "Normal"
print("\nOverall exact accuracy:", round(df["correct"].mean(), 3))
print("Attacks flagged as an attack:", round(df.loc[att, "detected"].mean(), 3))
print("Attacks that send an alert (High/Critical):", round(df.loc[att, "alert"].mean(), 3))
print("Normal events that send an alert:", round(df.loc[normal, "alert"].mean(), 3))
print("Engine time: median", round(df["ms"].iloc[1:].median()), "ms, 95th percentile",
      round(np.percentile(df["ms"].iloc[1:], 95)), "ms")