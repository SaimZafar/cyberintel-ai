import sys
import json
import time
from pathlib import Path

import pandas as pd
import shap
from fastapi import FastAPI, HTTPException

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

import explain                                                   # noqa: E402
from engine import Engine                                        # noqa: E402
from explanation import build_explanation, check_explanation    # noqa: E402
from features import CLASSES, CAT_COLS, LOG_COLS, make_features  # noqa: E402

explain.REF_PATH = str(BASE / "models" / "normal_reference.npz")

MODEL_VERSION = "v1"
NUMERIC_RAW = LOG_COLS + explain.PLAIN_COLS

engine = Engine(str(BASE / "models") + "/")
engine.iforest.n_jobs = 1
engine.clf.set_params(n_jobs=1)
explainer = shap.TreeExplainer(engine.clf)
reference = explain.load_reference()
config = json.load(open(BASE / "models" / "feature_config.json"))
cats, feature_names = config["cats"], config["feature_names"]

app = FastAPI(title="CyberIntel AI engine")


def to_frame(event):
    """Raw event dict -> one-row table. Missing values get a default and are reported."""
    row, missing = {}, []
    for c in NUMERIC_RAW:
        value = event.get(c)
        if value is None:
            row[c] = 0.0
            missing.append(c)
            continue
        try:
            row[c] = float(value)
        except (TypeError, ValueError):
            raise HTTPException(422, f"field '{c}' must be a number")
    for c in CAT_COLS:
        value = event.get(c)
        if value is None:
            row[c] = "-"
            missing.append(c)
        else:
            row[c] = str(value)
    return pd.DataFrame([row]), missing


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "n_features": len(feature_names)}


@app.post("/detect")
def detect(event: dict):
    start = time.time()
    raw_df, missing = to_frame(event)
    X = make_features(raw_df, cats)[feature_names]
    d = engine.decide(X).iloc[0]

    category = str(d["category"])
    confidence, anomaly = float(d["confidence"]), float(d["anomaly"])
    score, level = float(d["severity_score"]), str(d["severity_level"])
    incident = bool(d["alert"])
    name_conf = float(d["name_conf"])
    confident_name = name_conf >= 0.8
    family = "Exploit-type" if category in ("Exploits", "DoS", "Backdoor", "Analysis") else category

    top_features, steps = [], []
    if not incident:
        text = "Below the alert threshold. No incident was created."
    elif category == "Normal":
        e = build_explanation(category, confidence, anomaly, score, level, [])
        text, steps = e["text"], e["steps"]
    else:
        k = CLASSES.index(category)
        sv = explainer.shap_values(X)
        v = sv[k][0] if isinstance(sv, list) else sv[0, :, k]
        raw_row = raw_df.iloc[0]
        reasons = explain.top_reasons(feature_names, X.iloc[0].to_numpy(), raw_row, v, reference)
        args = (category, confidence, anomaly, score, level, reasons)
        e = build_explanation(*args)
        ok, _ = check_explanation(e["text"], *args)
        if ok:
            text = e["text"]
        else:
            text = (f"Classified as {category} with {confidence * 100:.0f}% confidence. "
                    f"Severity is {level} (score {score:.1f}).")
        steps = e["steps"]
        for r in reasons:
            value = raw_row[r["feature"]]
            top_features.append({
                "name": r["feature"],
                "value": value if isinstance(value, str) else float(value),
                "contribution": r["shap"],
                "direction": r["direction"],
                "text": r["text"],
            })
    if incident and not confident_name:
        article = "an" if family[0] in "AEIOU" else "a"
        text += (f" The specific attack type is uncertain. Treat this as {article} "
                 f"{family} attack until an analyst confirms it.")
    return {
        "incident": incident,
        "alert": incident,
        "p_attack": float(d["p_attack"]),
        "name_confidence": name_conf,
        "name_is_confident": confident_name,
        "attack_family": family,
        "attack_category": category,
        "confidence": confidence,
        "anomaly_score": anomaly,
        "severity_score": score,
        "severity_level": level,
        "severity_breakdown": {"confidence": float(d["C"]), "anomaly": anomaly,
                               "category_weight": float(d["W"])},
        "top_features": top_features,
        "explanation_text": text,
        "suggested_steps": steps,
        "explanation_source": "template",
        "model_version": MODEL_VERSION,
        "missing_fields": missing,
        "processing_ms": round((time.time() - start) * 1000, 1),
    }