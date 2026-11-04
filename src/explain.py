import os
import numpy as np
import pandas as pd
from features import CAT_COLS, LOG_COLS

PLAIN_COLS = ["swin", "dwin", "trans_depth", "ct_src_dport_ltm", "ct_dst_sport_ltm",
              "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports",
              "sttl", "dttl", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
              "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"]

NUMERIC = LOG_COLS + PLAIN_COLS
REF_PATH = "models/normal_reference.npz"

NICE = {
    "dur": "flow duration (seconds)", "spkts": "packets sent by the source",
    "dpkts": "packets sent by the destination", "sbytes": "bytes sent by the source",
    "dbytes": "bytes sent by the destination", "rate": "packet rate",
    "sload": "source load (bits per second)", "dload": "destination load (bits per second)",
    "sloss": "source packets lost or retransmitted", "dloss": "destination packets lost or retransmitted",
    "sinpkt": "time between source packets", "dinpkt": "time between destination packets",
    "sjit": "source jitter", "djit": "destination jitter",
    "stcpb": "source TCP sequence number", "dtcpb": "destination TCP sequence number",
    "tcprtt": "TCP round trip time", "synack": "TCP SYN to ACK time",
    "ackdat": "TCP ACK to data time", "smean": "average size of source packets",
    "dmean": "average size of destination packets", "swin": "source TCP window",
    "dwin": "destination TCP window", "trans_depth": "HTTP transaction depth",
    "response_body_len": "HTTP response size",
    "ct_src_dport_ltm": "recent connections from the same source to the same port",
    "ct_dst_sport_ltm": "recent connections to the same destination from the same source port",
    "is_ftp_login": "FTP login flag", "ct_ftp_cmd": "number of FTP commands",
    "ct_flw_http_mthd": "number of HTTP methods",
    "is_sm_ips_ports": "same address and port on both sides flag",
    "sttl": "source time to live (TTL)", "dttl": "destination time to live (TTL)",
    "ct_srv_src": "recent connections with the same service and source address",
    "ct_state_ttl": "recent connections with the same state and TTL values",
    "ct_dst_ltm": "recent connections to the same destination address",
    "ct_dst_src_ltm": "recent connections with the same source and destination address",
    "ct_src_ltm": "recent connections from the same source address",
    "ct_srv_dst": "recent connections with the same service and destination address",
}
GROUP_NAME = {"proto": "protocol", "service": "service", "state": "connection state"}


def build_reference(n=5000):
    fit = pd.read_parquet("data/processed/fit.parquet")
    normal = fit[fit["attack_cat"] == "Normal"]
    normal = normal.sample(min(n, len(normal)), random_state=0)
    np.savez(REF_PATH, **{c: np.sort(normal[c].to_numpy(dtype=float)) for c in NUMERIC})


def load_reference():
    if not os.path.exists(REF_PATH):
        build_reference()
    z = np.load(REF_PATH)
    return {c: z[c] for c in z.files}


def fmt(v):
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.3g}"


def pct(share):
    """Whole percent, never shown as 100%."""
    return min(int(share * 100), 99)


def numeric_text(col, v, ref):
    r = ref[col]
    below = np.searchsorted(r, v, side="left") / len(r)       # share of normal flows strictly lower
    above = 1 - np.searchsorted(r, v, side="right") / len(r)  # share of normal flows strictly higher
    name = NICE.get(col, col)
    if below >= 0.90:
        return f"{name} is {fmt(v)}, higher than {pct(below)}% of normal traffic"
    if above >= 0.90:
        return f"{name} is {fmt(v)}, lower than {pct(above)}% of normal traffic"
    return f"{name} is {fmt(v)}, within the usual range of normal traffic"


def category_text(group, actual, kept):
    label = GROUP_NAME[group]
    if group == "service" and actual == "-":
        return "no recognised service was used"
    if actual not in kept:
        return f"the {label} is {actual}, an uncommon value"
    return f"the {label} is {actual}"


def top_reasons(columns, x_row, raw_row, shap_row, ref, k=5):
    """The k features that moved the decision most, with plain-language text.
    The 0/1 columns of one group (for example all service_* columns) are merged into one reason."""
    items = []
    for g in CAT_COLS:
        idx = [j for j, c in enumerate(columns) if c.startswith(g + "_")]
        kept = [columns[j][len(g) + 1:] for j in idx if columns[j] != f"{g}_other"]
        items.append({"name": g, "shap": float(sum(shap_row[j] for j in idx)), "kept": kept})
    for j, c in enumerate(columns):
        if not any(c.startswith(g + "_") for g in CAT_COLS):
            items.append({"name": c, "shap": float(shap_row[j]), "kept": None})

    items.sort(key=lambda t: -abs(t["shap"]))
    reasons = []
    for it in items[:k]:
        if it["kept"] is not None:
            text = category_text(it["name"], raw_row[it["name"]], it["kept"])
        else:
            text = numeric_text(it["name"], float(raw_row[it["name"]]), ref)
        reasons.append({"feature": it["name"], "text": text, "shap": it["shap"],
                        "direction": "supports" if it["shap"] > 0 else "argues against"})
    return reasons


if __name__ == "__main__":
    import joblib
    import shap
    from features import CLASSES

    D = "data/processed/"
    model = joblib.load("models/classifier_v1.joblib")
    Xv = pd.read_parquet(D + "X_val.parquet")
    raw = pd.read_parquet(D + "val.parquet")
    yv = pd.read_parquet(D + "y_val.parquet")["y"].to_numpy()

    pos = np.random.RandomState(1).choice(len(Xv), 500, replace=False)
    S, R, yS = Xv.iloc[pos], raw.iloc[pos], yv[pos]
    pred = model.predict(S)
    sv = shap.TreeExplainer(model).shap_values(S)
    ref = load_reference()

    shown = 0
    for i in range(len(S)):
        if pred[i] == yS[i] and pred[i] != 0 and shown < 6:
            v = sv[pred[i]][i] if isinstance(sv, list) else sv[i, :, pred[i]]
            print(f"\nevent {i}: true = {CLASSES[yS[i]]}, predicted = {CLASSES[pred[i]]}")
            for r in top_reasons(list(S.columns), S.iloc[i].to_numpy(), R.iloc[i], v, ref):
                print(f"   [{r['direction']:<13}] {r['text']}   (shap {r['shap']:+.2f})")
            shown += 1