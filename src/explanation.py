import re
import numpy as np
import pandas as pd
from explain import top_reasons, load_reference, pct
from features import CLASSES

CLASS_INFO = {
    "Generic": "Generic attacks target cryptographic weaknesses and work against any block cipher, whatever its settings.",
    "Exploits": "Exploits abuse a known flaw or bug in software to gain control or run unwanted actions.",
    "Fuzzers": "Fuzzers send large amounts of random or malformed data to find bugs or crash a program.",
    "DoS": "Denial of service attacks try to make a server or network resource unavailable to legitimate users.",
    "Reconnaissance": "Reconnaissance gathers information about a network and its hosts, for example by probing for open ports and services.",
    "Analysis": "Analysis attacks include port scans, spam and probing of web scripts to learn about a target.",
    "Backdoor": "A backdoor is a hidden way to bypass normal authentication and keep unauthorised remote access.",
    "Shellcode": "Shellcode is a small piece of code used as the payload when exploiting a vulnerability to take control of a process.",
    "Worms": "Worms copy themselves and spread to other computers over the network.",
}

STEPS = {
    "Generic": ["Check which hosts and services are involved and whether the activity is expected",
                "Review recent connections from the same source",
                "Look for repeated identical requests that may indicate automated abuse"],
    "Exploits": ["Identify the targeted service and check that it is patched",
                 "Review logs of the destination host around this time",
                 "Check for follow-up activity from the same source"],
    "Fuzzers": ["Find the target service and look for crashes or restarts",
                "Check the source for repeated malformed requests",
                "Consider rate limiting the source after review"],
    "DoS": ["Check traffic volume to the target and the availability of the service",
            "Identify the main sources of the traffic",
            "Prepare rate limits or filtering for the analyst to approve"],
    "Reconnaissance": ["Find the ports and hosts the source touched",
                       "Check whether the source is known or internal",
                       "Watch for later attacks from the same source"],
    "Analysis": ["Review the scanned ports and scripts that were probed",
                 "Check whether the source is a known scanner or tester",
                 "Watch for follow-up exploitation attempts"],
    "Backdoor": ["Check the destination host for unknown listening services or accounts",
                 "Review remote access logs",
                 "Isolate the host if the analyst confirms a compromise"],
    "Shellcode": ["Check the target process and host for unexpected behaviour",
                  "Look for the exploit that delivered the code",
                  "Isolate the host if the analyst confirms a compromise"],
    "Worms": ["Find other hosts contacted by the same source",
              "Check whether similar traffic is spreading across the network",
              "Isolate the affected hosts if the analyst confirms it"],
    "Normal": ["Compare the flow with the usual traffic of this service",
               "Check whether a new device or application explains it",
               "Mark it as a false alarm if it is expected"],
}

NUM = re.compile(r"\d[\d,]*\.?\d*")


def build_explanation(category, confidence, anomaly, score, level, reasons):
    if category == "Normal":
        text = (f"The classifier sees normal traffic, but this flow is more unusual than "
                f"{pct(anomaly)}% of normal traffic. Severity is {level} (score {score:.1f}). "
                "Treat it as unknown behaviour and compare it with the usual traffic of this service.")
        return {"text": text, "steps": STEPS["Normal"]}

    supports = [r for r in reasons if r["direction"] == "supports"][:3]
    parts = [f"The engine classified this flow as {category} with {confidence * 100:.0f}% confidence. "
             f"Severity is {level} (score {score:.1f}).",
             CLASS_INFO[category]]
    if supports:
        parts.append("Main reasons: " + "; ".join(r["text"] for r in supports) + ".")
        if all("within the usual range" in r["text"] for r in supports):
            parts.append("These values look ordinary on their own. The model relies on combinations "
                         "of values, so treat this as a lead and verify it.")
    if anomaly >= 0.8:
        parts.append(f"The flow is also more unusual than {pct(anomaly)}% of normal traffic.")
    parts.append("These reasons show what the model relied on, not proof of malicious intent.")
    return {"text": " ".join(parts), "steps": STEPS[category]}


def check_explanation(text, category, confidence, anomaly, score, level, reasons):
    """Every number must come from the evidence; no other attack class may be named."""
    evidence = " ".join([f"{confidence * 100:.0f}", f"{score:.1f}", str(pct(anomaly))]
                        + [r["text"] for r in reasons])
    allowed = {n.rstrip(".,") for n in NUM.findall(evidence)}
    bad = [n.rstrip(".,") for n in NUM.findall(text) if n.rstrip(".,") not in allowed]
    other = [c for c in CLASSES if c != category and re.search(r"\b" + c + r"\b", text)]
    missing = [] if (category == "Normal" or category in text) else ["category"]
    if level not in text:
        missing.append("level")
    ok = not bad and not other and not missing
    return ok, {"bad_numbers": bad, "other_classes": other, "missing": missing}


if __name__ == "__main__":
    import shap
    from engine import Engine

    D = "data/processed/"
    eng = Engine()
    Xv = pd.read_parquet(D + "X_val.parquet")
    raw = pd.read_parquet(D + "val.parquet")

    pos = np.random.RandomState(1).choice(len(Xv), 500, replace=False)
    S, R = Xv.iloc[pos], raw.iloc[pos]
    dec = eng.decide(S)
    sv = shap.TreeExplainer(eng.clf).shap_values(S)
    ref = load_reference()
    cols = list(S.columns)

    passed = total = shown = 0
    last = None
    for i in range(len(S)):
        d = dec.iloc[i]
        if d["category"] == "Normal":
            continue
        k = CLASSES.index(d["category"])
        v = sv[k][i] if isinstance(sv, list) else sv[i, :, k]
        reasons = top_reasons(cols, S.iloc[i].to_numpy(), R.iloc[i], v, ref)
        args = (d["category"], d["confidence"], d["anomaly"], d["severity_score"], d["severity_level"], reasons)
        e = build_explanation(*args)
        ok, info = check_explanation(e["text"], *args)
        total += 1
        passed += int(ok)
        last = (e, args)
        if shown < 3:
            print(f"\n--- event {i}: {d['category']} ---")
            print(e["text"])
            for step in e["steps"]:
                print("  -", step)
            print("check passed:", ok)
            shown += 1

    print(f"\nexplanations that passed the check: {passed} of {total}")

    # tamper test: the checker must reject wrong texts
    e, args = last 
    conf_pct = f"{args[1] * 100:.0f}%"
    tests = {
        "changed confidence": e["text"].replace(conf_pct, "12%", 1),
        "invented number": e["text"] + " It affected 73 hosts.",
        "wrong attack named": e["text"] + " This is a Worms outbreak.",
    }
    print("\nTAMPER TEST (each should be rejected, so it should print False)")
    for name, bad_text in tests.items():
        print(f"  {name:<20}", check_explanation(bad_text, *args)[0])