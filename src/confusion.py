import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from features import CLASSES

D = "data/processed/"
name = json.load(open("models/best_model.json"))["name"]
model = joblib.load(f"models/best_{name}.joblib")

Xv = pd.read_parquet(D + "X_val.parquet")
yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()
pred = model.predict(Xv)

labels = list(range(len(CLASSES)))
cm = confusion_matrix(yv, pred, labels=labels)
cm_norm = cm / cm.sum(axis=1, keepdims=True)   # each row adds up to 1

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Validation confusion matrix ({name}), rows add up to 1")
plt.tight_layout()
plt.savefig("reports/figures/confusion_val_best.png", dpi=150)
print("saved reports/figures/confusion_val_best.png\n")

print("Biggest mistakes (true class -> predicted as):")
pairs = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in labels for j in labels if i != j]
for count, true, predicted in sorted(pairs, reverse=True)[:12]:
    print(f"{count:>6}  {true:<15} -> {predicted}")

print("\nWhere do the weak classes go?")
for c in ["DoS", "Analysis", "Backdoor"]:
    i = CLASSES.index(c)
    top = np.argsort(-cm_norm[i])[:3]
    print(c, "->", {CLASSES[j]: round(float(cm_norm[i, j]), 2) for j in top})