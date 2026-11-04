import json
import subprocess
import sys
import time
from pathlib import Path

STEPS = ["make_splits", "prepare_data", "baseline_lr", "train_models", "weight_strength",
         "dedup_test", "tune_xgb", "confusion", "anomaly", "SET_THRESHOLD",
         "evaluate_engine", "ablate_severity", "eval_demo_pool", "profile_engine"]

start_from = sys.argv[1] if len(sys.argv) > 1 else STEPS[0]   # e.g. python src\run_all.py anomaly
Path("reports/logs").mkdir(parents=True, exist_ok=True)
t_all = time.time()

for step in STEPS[STEPS.index(start_from):]:
    if step == "SET_THRESHOLD":
        p = "models/anomaly_config.json"
        c = json.load(open(p))
        c["threshold"] = 0.99
        json.dump(c, open(p, "w"))
        print("\nanomaly threshold set to 0.99")
        continue

    print(f"\n=== {step} ===", flush=True)
    t = time.time()
    with open(f"reports/logs/{step}.txt", "w") as log:
        proc = subprocess.Popen([sys.executable, f"src/{step}.py"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
    print(f"[{step} finished in {time.time() - t:.0f} s, exit code {proc.returncode}]")
    if proc.returncode != 0:
        print("STOPPED. Fix the error above, then restart from this step:  python src\\run_all.py", step)
        break

print(f"\nTotal time: {(time.time() - t_all) / 60:.1f} minutes")