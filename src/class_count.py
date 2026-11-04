from pathlib import Path
import pandas as pd
from features import CLASSES

RAW, D = "data/raw/", "data/processed/"
sources = {
    "official train": (RAW + "UNSW_NB15_training-set.csv", "csv"),
    "official test": (RAW + "UNSW_NB15_testing-set.csv", "csv"),
    "fit": (D + "fit.parquet", "parquet"),
    "val": (D + "val.parquet", "parquet"),
    "test_eval": (D + "test_eval.parquet", "parquet"),
    "demo_pool": (D + "demo_pool.parquet", "parquet"),
}

counts = {}
for name, (path, kind) in sources.items():
    if not Path(path).exists():
        print(f"skipped (file not found): {path}")
        continue
    col = pd.read_csv(path, usecols=["attack_cat"])["attack_cat"] if kind == "csv" \
        else pd.read_parquet(path, columns=["attack_cat"])["attack_cat"]
    counts[name] = col.astype(str).str.strip().value_counts()

table = pd.DataFrame(counts).reindex(CLASSES).fillna(0).astype(int)
table.loc["TOTAL"] = table.sum()

pd.set_option("display.width", 200)
print("\nROWS PER CLASS")
print(table.to_string())

pct = (table.drop(index="TOTAL") / table.loc["TOTAL"] * 100).round(2)
print("\nPERCENT OF EACH SET")
print(pct.to_string())

table.to_csv("reports/class_counts.csv")
print("\nsaved reports/class_counts.csv")