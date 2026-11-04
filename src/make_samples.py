import json, urllib.request

URL = "http://127.0.0.1:8000/detect"
events = [json.loads(l) for l in open("replay_events.jsonl")]
labels = {json.loads(l)["event_id"]: json.loads(l)["true_class"] for l in open("replay_labels.jsonl")}

samples, seen = {}, set()
for e in events:
    cls = labels[e["event_id"]]
    if cls in seen:
        continue
    req = urllib.request.Request(URL, data=json.dumps(e).encode(),
                                 headers={"Content-Type": "application/json"})
    samples[cls] = json.load(urllib.request.urlopen(req))
    seen.add(cls)

json.dump(samples, open("sample_responses.json", "w"), indent=2)
print({k: (v["alert"], v["attack_category"]) for k, v in samples.items()})