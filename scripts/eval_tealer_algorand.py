"""Baseline: run Tealer (crytic/tealer), a real open-source static analyzer for
TEAL/PyTeal, against the 45-contract Algorand test set, and score it with the
same metric definitions used for every LLM configuration in this study.

Tealer is deterministic (no sampling), so this produces a single point estimate
rather than a mean +/- std -- there is no seed to vary.
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
OUT_DIR = os.path.join(BASE_DIR, "results_multisample", "algorand_tealer_baseline")
TEALER_PYTHON = os.path.join(BASE_DIR, ".venv-tealer", "bin", "python3")
TEALER_BIN = os.path.join(BASE_DIR, ".venv-tealer", "bin", "tealer")

MAP_TRUTH_TO_DISPLAY = {
    "transaction_fee": "Unchecked Transaction Fee",
    "close_remainder_to": "Unchecked Close Remainder To",
    "asset_close_to": "Unchecked Asset Close To",
    "rekey_to": "Unchecked Rekey to",
    "arbitrary_delete": "Arbitrary delete",
    "arbitrary_update": "Arbitrary update",
    "Unchecked_Asset_Receiver": "Unchecked Asset Receiver",
    "Unchecked_Payment_Receiver": "Unchecked Payment Receiver",
    "no vuln": "Not Vulnerable",
}
SAFE_LABELS = {"Not Vulnerable"}

# Tealer detector -> this study's category. Detectors with no mapping
# (group-size-check, constant-gtxn, self-access, sender-access) are excluded:
# they don't correspond to any category in this taxonomy and are ignored
# rather than force-mapped, so they can never contribute a spurious FP here.
DETECTOR_TO_CATEGORY = {
    "unprotected-deletable": "Arbitrary delete",
    "is-deletable": "Arbitrary delete",
    "unprotected-updatable": "Arbitrary update",
    "is-updatable": "Arbitrary update",
    "can-close-account": "Unchecked Close Remainder To",
    "can-close-asset": "Unchecked Asset Close To",
    "rekey-to": "Unchecked Rekey to",
    "missing-fee-check": "Unchecked Transaction Fee",
}

# Ground-truth categories with no corresponding Tealer detector: Tealer can
# never predict these, by construction of its fixed rule set.
UNCOVERED_CATEGORIES = {"Unchecked Asset Receiver", "Unchecked Payment Receiver"}


def compile_to_teal(pyteal_source, workdir):
    py_path = os.path.join(workdir, "contract.py")
    teal_path = os.path.join(workdir, "contract.teal")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(pyteal_source)
    try:
        result = subprocess.run(
            [TEALER_PYTHON, py_path], capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if result.returncode != 0 or not result.stdout.strip():
        return None, result.stderr.strip()[:500]
    with open(teal_path, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    return teal_path, None


def run_tealer(teal_path, workdir):
    json_path = os.path.join(workdir, "result.json")
    try:
        subprocess.run(
            [TEALER_BIN, "--json", json_path, "detect", "--contracts", teal_path],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if not os.path.exists(json_path):
        return None, "tealer produced no output"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data, None


def predicted_categories(tealer_json):
    preds = set()
    for entry in tealer_json.get("result", []):
        check = entry.get("check")
        count = entry.get("count", 0)
        if count and check in DETECTOR_TO_CATEGORY:
            preds.add(DETECTOR_TO_CATEGORY[check])
    return preds


def compute_metrics(results):
    tp = fp = tn = fn = 0
    per_vuln = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        truth = r["ground_truth_display"]
        preds = set(r["predictions"])
        preds_is_safe = len(preds) == 0
        if truth == "Not Vulnerable":
            if preds_is_safe:
                tn += 1
            else:
                fp += len(preds)
                for p in preds:
                    per_vuln[p]["fp"] += 1
        else:
            if truth in preds:
                tp += 1
                per_vuln[truth]["tp"] += 1
                extras = preds - {truth}
                fp += len(extras)
                for e in extras:
                    per_vuln[e]["fp"] += 1
            else:
                fn += 1
                per_vuln[truth]["fn"] += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1_score": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_vulnerability": dict(per_vuln),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    test_files = sorted(
        os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".json")
    )
    results = []
    for path in test_files:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        truth_display = MAP_TRUTH_TO_DISPLAY.get(d["vulnerability"], d["vulnerability"])
        with tempfile.TemporaryDirectory() as workdir:
            teal_path, err = compile_to_teal(d["smart_contract"], workdir)
            if teal_path is None:
                print(f"  [{os.path.basename(path)}] COMPILE FAILED: {err}")
                results.append({
                    "file": os.path.basename(path), "ground_truth": d["vulnerability"],
                    "ground_truth_display": truth_display, "predictions": [],
                    "error": f"compile_failed: {err}",
                })
                continue
            tealer_json, err = run_tealer(teal_path, workdir)
            if tealer_json is None:
                print(f"  [{os.path.basename(path)}] TEALER FAILED: {err}")
                results.append({
                    "file": os.path.basename(path), "ground_truth": d["vulnerability"],
                    "ground_truth_display": truth_display, "predictions": [],
                    "error": f"tealer_failed: {err}",
                })
                continue
            preds = sorted(predicted_categories(tealer_json))
        print(f"  [{os.path.basename(path)}] truth={truth_display} | pred={preds}")
        results.append({
            "file": os.path.basename(path), "ground_truth": d["vulnerability"],
            "ground_truth_display": truth_display, "predictions": preds,
        })

    metrics = compute_metrics(results)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_samples": 1, "note": "deterministic tool, single run", "metrics": metrics}, f, indent=2)

    print("\nSUMMARY (Tealer, Algorand, 45-contract test set):")
    for k, v in metrics.items():
        if k in ("accuracy", "precision", "recall", "f1_score"):
            print(f"  {k:10s}: {v}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"  uncovered categories (no Tealer detector exists): {UNCOVERED_CATEGORIES}")


if __name__ == "__main__":
    main()
