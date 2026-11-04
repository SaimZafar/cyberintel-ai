import importlib
import json
import os
import sys
import time
from pathlib import Path

root = Path(".")
print("python:", sys.version.split()[0])
for mod in ["pandas", "numpy", "sklearn", "xgboost", "shap", "fastapi", "lightgbm", "torch", "pytorch_tabnet"]:
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:<16} {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"  {mod:<16} NOT AVAILABLE ({type(e).__name__})")

print("\nkey artifacts")
for p in ["models/classifier_v1.joblib", "models/classifier_v1.json", "models/anomaly_iforest.joblib",
          "models/anomaly_reference.npy", "models/anomaly_config.json", "models/alert_config.json",
          "models/feature_config.json", "models/normal_reference.npz", "service/main.py", "src/engine.py",
          "data/processed/fit.parquet", "data/processed/val.parquet", "data/processed/test_eval.parquet",
          "data/processed/demo_pool.parquet", "reports/final_test_results.csv"]:
    f = root / p
    print(f"  {'OK     ' if f.exists() else 'MISSING'} {p}" +
          (f"  ({f.stat().st_size / 1e6:.1f} MB, {time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))})"
           if f.exists() else ""))

for p, key in [("models/anomaly_config.json", None), ("models/alert_config.json", None),
               ("models/classifier_v1.json", None)]:
    f = root / p
    if f.exists():
        print(f"\n{p}:", json.load(open(f)))

fc = root / "models/feature_config.json"
if fc.exists():
    c = json.load(open(fc))
    print("\nfeature_config: n_features =", len(c["feature_names"]), "| categories:", {k: len(v) for k, v in c["cats"].items()})

print("\nfiles in src/ (newest first)")
for f in sorted((root / "src").glob("*.py"), key=os.path.getmtime, reverse=True):
    print(f"  {time.strftime('%m-%d %H:%M', time.localtime(f.stat().st_mtime))}  {f.name}")

print("\nstring checks in service/main.py")
s = (root / "service/main.py").read_text() if (root / "service/main.py").exists() else ""
for k in ['d["alert"]', "name_conf", "n_jobs=1", '"alert"']:
    print(f"  {k:<14} {'found' if k in s else 'NOT FOUND'}")