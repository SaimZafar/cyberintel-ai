import json
import time
import urllib.request
import numpy as np
import pandas as pd

URL = "http://127.0.0.1:8000/detect"
demo = pd.read_parquet("data/processed/demo_pool.parquet")


def call(row):
    event = json.loads(row.drop(["attack_cat", "label", "group"]).to_json())
    req = urllib.request.Request(URL, data=json.dumps(event).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    res = json.load(urllib.request.urlopen(req))
    return res, (time.time() - t) * 1000


print(f"{'true':<15}{'predicted':<16}{'conf':>6}{'anomaly':>9}{'severity':>10}{'score':>7}{'engine ms':>11}")
shown_text = False
for _, row in demo.groupby("attack_cat").head(1).iterrows():
    res, ms = call(row)
    print(f"{row['attack_cat']:<15}{res['attack_category']:<16}{res['confidence']:6.2f}"
          f"{res['anomaly_score']:9.2f}{res['severity_level']:>10}{res['severity_score']:7.1f}"
          f"{res['processing_ms']:11.0f}")
    if res["attack_category"] != "Normal" and not shown_text:
        print("\nExample explanation:\n", res["explanation_text"], "\n", res["suggested_steps"], "\n")
        shown_text = True

times, engine_times = [], []
for _, row in demo.sample(min(100, len(demo)), random_state=0).iterrows():
    res, ms = call(row)
    times.append(ms)
    engine_times.append(res["processing_ms"])
print(f"\n100 requests: round trip median {np.median(times):.0f} ms, 95th percentile {np.percentile(times, 95):.0f} ms")
print(f"engine time inside the service: median {np.median(engine_times):.0f} ms, 95th percentile {np.percentile(engine_times, 95):.0f} ms")