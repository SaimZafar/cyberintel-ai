import json
import os
import joblib
import numpy as np
import pandas as pd
from features import CLASSES

# how dangerous each attack type is
ATTACK_WEIGHTS = {"Generic": 0.70, "Exploits": 0.90, "Fuzzers": 0.50, "DoS": 0.90,
                  "Reconnaissance": 0.50, "Analysis": 0.60, "Backdoor": 0.95,
                  "Shellcode": 0.95, "Worms": 1.0}
DEFAULT_WEIGHT = 0.50   # used when the classifier says Normal


class Engine:
    def __init__(self, models_dir="models/"):
        self.clf = joblib.load(models_dir + "classifier_v1.joblib")
        self.iforest = joblib.load(models_dir + "anomaly_iforest.joblib")
        self.ref = np.load(models_dir + "anomaly_reference.npy")
        self.threshold = json.load(open(models_dir + "anomaly_config.json"))["threshold"]
        cfg = models_dir + "alert_config.json"
        self.alert_threshold = (json.load(open(cfg))["threshold"]
                                if os.path.exists(cfg) else 0.956)

    def anomaly(self, X):
        s = self.iforest.score_samples(X)
        return (len(self.ref) - np.searchsorted(self.ref, s, side="right")) / len(self.ref)

    def decide(self, X, weights=(0.4, 0.3, 0.3), threshold=None):
        threshold = self.threshold if threshold is None else threshold
        wc, wa, ww = weights
        proba = self.clf.predict_proba(X)
        pred = proba.argmax(axis=1)
        a = self.anomaly(X)
        confidence = proba.max(axis=1)
        is_attack = pred != 0

        # the alert decision: how likely is it that this is any attack at all
        p_attack = 1.0 - proba[:, 0]
        alert = p_attack > self.alert_threshold
        name_conf = proba[:, 1:].max(axis=1) / np.maximum(p_attack, 1e-9)
        # C = how sure the classifier is that this is an attack
        C = np.where(is_attack, confidence, 1 - proba[:, 0])
        W = np.array([ATTACK_WEIGHTS.get(CLASSES[k], DEFAULT_WEIGHT) for k in pred])
        score = 100 * (wc * C + wa * a + ww * W)
        level = pd.cut(score, bins=[-np.inf, 40, 65, 85, np.inf],
                       labels=["Low", "Medium", "High", "Critical"], right=False)

        unusual = (~is_attack) & (a > threshold)   # kept only so old scripts still run
        return pd.DataFrame({
            "category": [CLASSES[k] for k in pred],
            "confidence": confidence,
            "C": C,
            "W": W,
            "anomaly": a,
            "p_attack": p_attack,
            "name_conf": name_conf,
            "alert": alert,
            "severity_score": score.round(1),
            "severity_level": level.astype(str),
            "incident": alert,
            "anomaly_only": unusual,
        })