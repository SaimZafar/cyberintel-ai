import json
import pandas as pd

D = "data/processed/"
demo = pd.read_parquet(D + "demo_pool.parquet")

# classes that alert reliably, plus Normal for the quiet scenario
KEEP = ["Exploits", "Reconnaissance", "Backdoor", "Shellcode", "DoS", "Normal"]
demo = demo[demo["attack_cat"].isin(KEEP)]

parts = []
for cls in KEEP:
    g = demo[demo["attack_cat"] == cls]
    parts.append(g.sample(min(25, len(g)), random_state=42))
demo = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
demo["event_id"] = [f"evt-{i:04d}" for i in range(len(demo))]

true_class = demo["attack_cat"].tolist()
drop = ["attack_cat", "label", "group", "id"]
events = demo.drop(columns=[c for c in drop if c in demo.columns])

with open("replay_events.jsonl", "w") as fe, open("replay_labels.jsonl", "w") as fl:
    for i in range(len(events)):
        rec = json.loads(events.iloc[i].to_json())
        fe.write(json.dumps(rec) + "\n")
        fl.write(json.dumps({"event_id": rec["event_id"], "true_class": true_class[i]}) + "\n")

print(len(demo), "events written")
print(demo["attack_cat"].value_counts().to_dict())