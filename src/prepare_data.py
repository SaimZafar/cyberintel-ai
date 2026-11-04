import json
import pandas as pd
from features import learn_categories, make_features, get_labels, LOG_COLS

D = "data/processed/"
names = ["fit", "val", "test_eval", "demo_pool"]
data = {n: pd.read_parquet(D + n + ".parquet") for n in names}

print("log columns with negative values:",
      [c for c in LOG_COLS if data["fit"][c].min() < 0])

# learn the categories from the training part only, then reuse everywhere
cats = learn_categories(data["fit"])
feature_names = make_features(data["fit"], cats).columns.tolist()

with open("models/feature_config.json", "w") as f:
    json.dump({"cats": cats, "feature_names": feature_names}, f, indent=2)

print("categories kept:", cats)
print("number of features:", len(feature_names))

for n, df in data.items():
    X = make_features(df, cats)[feature_names]
    y, y_bin = get_labels(df)
    X.to_parquet(D + f"X_{n}.parquet")
    pd.DataFrame({"y": y, "y_bin": y_bin}).to_parquet(D + f"y_{n}.parquet")
    print(n, X.shape, "NaN:", int(X.isna().sum().sum()))