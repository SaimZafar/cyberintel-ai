import json
import urllib.request
import pandas as pd

demo = pd.read_parquet("data/processed/demo_pool.parquet")
for cls in ["DoS", "Backdoor", "Generic", "Normal"]:
    row = demo[demo["attack_cat"] == cls].iloc[3]
    event = json.loads(row.drop(["attack_cat", "label", "group"]).to_json())
    req = urllib.request.Request("http://127.0.0.1:8000/detect", data=json.dumps(event).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    print(f"true {cls:<9} alert={r['alert']!s:<6} category={r['attack_category']:<10} "
          f"family={r['attack_family']:<13} name_conf={r['name_confidence']:.2f} "
          f"confident={r['name_is_confident']}")
    if r["alert"]:
        print("   ", r["explanation_text"][-140:])