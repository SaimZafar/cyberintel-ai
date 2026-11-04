import json, urllib.request
import pandas as pd

URL = "http://127.0.0.1:8000/detect"
events = [json.loads(l) for l in open("replay_events.jsonl")]
labels = {json.loads(l)["event_id"]: json.loads(l)["true_class"] for l in open("replay_labels.jsonl")}

rows = []
for e in events:
    req = urllib.request.Request(URL, data=json.dumps(e).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    rows.append({"true": labels[e["event_id"]], "alert": bool(r["alert"]),
                 "category": r["attack_category"],
                 "family": r.get("attack_family"),
                 "confident": bool(r.get("name_is_confident")),
                 "ms": r["processing_ms"]})

df = pd.DataFrame(rows)
df["name_correct"] = df["category"] == df["true"]
out = df.groupby("true").agg(events=("alert", "size"), alert_rate=("alert", "mean"),
                             name_correct=("name_correct", "mean"),
                             confident=("confident", "mean")).round(2)
print(out.to_string())
print("\nmedian engine ms:", df["ms"].median())
df.to_csv("replay_results.csv", index=False)