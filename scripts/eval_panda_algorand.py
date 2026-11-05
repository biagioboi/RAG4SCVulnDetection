"""Baseline: run Panda (https://github.com/0xZhiyuan/Panda), a symbolic-execution
static analyzer for TEAL/PyTeal, against the 45-contract Algorand test set, scored
with the same metric definitions used for every LLM configuration in this study.

Panda is deterministic (Z3-backed symbolic execution, no sampling), so this
produces a single point estimate. Rule set "rule2" is used: unlike "rule1", it
covers all 6 of the 8 taxonomy categories Panda can structurally detect
(it has no detector for arbitrary categories outside its fixed rule set, same
caveat as Tealer).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
TEST_DIR = os.path.join(BASE_DIR, "test_contracts")
OUT_DIR = os.path.join(BASE_DIR, "results_multisample", "algorand_panda_baseline")
TEALER_PYTHON = os.path.join(BASE_DIR, ".venv-tealer", "bin", "python3")  # has pyteal
PANDA_PYTHON = os.path.join(BASE_DIR, ".venv-panda", "bin", "python3")
PANDA_REPO = "/tmp/panda_repo"
RULE_SET = "rule2"
GLOBAL_TIMEOUT_S = 25
Z3_TIMEOUT_MS = 8000
SUBPROCESS_TIMEOUT_S = 45

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

OUTPUT_PHRASE_TO_CATEGORY = {
    "Found an arbitrary update vulnerability": "Arbitrary update",
    "Found an arbitrary deletion vulnerability": "Arbitrary delete",
    "Found an unchecked payment receiver vulnerability": "Unchecked Payment Receiver",
    "Found an unchecked asset receiver vulnerability": "Unchecked Asset Receiver",
    "Found unchecked rekey-to parameter": "Unchecked Rekey to",
    "Found unchecked close-remainder-to parameter": "Unchecked Close Remainder To",
    "Found unchecked asset-close-to parameter": "Unchecked Asset Close To",
    "Found an unchecked transaction fee vulnerability": "Unchecked Transaction Fee",
}


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


def detect_mode(pyteal_source):
    if "Mode.Signature" in pyteal_source:
        return "-lsig"
    if "Mode.Application" in pyteal_source:
        return "-sc"
    return None


def run_panda(teal_path, mode_flag):
    cmd = [
        PANDA_PYTHON, "panda.py", mode_flag, "-s", teal_path,
        "-rs", RULE_SET, "-gt", str(GLOBAL_TIMEOUT_S), "-zt", str(Z3_TIMEOUT_MS),
    ]
    try:
        result = subprocess.run(
            cmd, cwd=PANDA_REPO, capture_output=True, text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, "subprocess_timeout"
    out = result.stdout + result.stderr
    if "ERROR:symExec" in out or "Traceback" in out:
        first_error = next((l for l in out.splitlines() if "ERROR" in l or "Error" in l), out[:200])
        return None, first_error
    return out, None


def predicted_categories(panda_output):
    preds = set()
    for phrase, category in OUTPUT_PHRASE_TO_CATEGORY.items():
        if phrase in panda_output:
            preds.add(category)
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
        mode_flag = detect_mode(d["smart_contract"])
        base = {
            "file": os.path.basename(path), "ground_truth": d["vulnerability"],
            "ground_truth_display": truth_display,
        }
        if mode_flag is None:
            print(f"  [{base['file']}] SKIP: could not detect Mode (Signature/Application)")
            results.append({**base, "predictions": [], "error": "unknown_mode"})
            continue
        with tempfile.TemporaryDirectory() as workdir:
            teal_path, err = compile_to_teal(d["smart_contract"], workdir)
            if teal_path is None:
                print(f"  [{base['file']}] COMPILE FAILED: {err}")
                results.append({**base, "predictions": [], "error": f"compile_failed: {err}"})
                continue
            panda_output, err = run_panda(teal_path, mode_flag)
            if panda_output is None:
                print(f"  [{base['file']}] PANDA FAILED: {err}")
                results.append({**base, "predictions": [], "error": f"panda_failed: {err}"})
                continue
            preds = sorted(predicted_categories(panda_output))
        print(f"  [{base['file']}] mode={mode_flag} truth={truth_display} | pred={preds}")
        results.append({**base, "predictions": preds})

    metrics = compute_metrics(results)
    n_errors = sum(1 for r in results if "error" in r)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "n_errors": n_errors, "details": results}, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_samples": 1, "note": "deterministic tool, single run", "n_errors": n_errors, "metrics": metrics}, f, indent=2)

    print(f"\nSUMMARY (Panda/{RULE_SET}, Algorand, 45-contract test set):")
    for k in ("accuracy", "precision", "recall", "f1_score"):
        print(f"  {k:10s}: {metrics[k]}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"  parse/analysis failures: {n_errors}/{len(results)}")


if __name__ == "__main__":
    main()
