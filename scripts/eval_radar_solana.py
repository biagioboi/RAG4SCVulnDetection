"""Score Radar (https://github.com/Auditware/radar), an open-source, actively
maintained static analyzer for Anchor/Rust smart contracts, against the
97-contract Solana test set, using the same metric definitions as every other
Solana configuration in this study.

This script only scores Radar's output.json (already produced by running
`./radar -p /tmp/solana_radar_input` from the cloned Auditware/radar repo
against the test set exported to individual .rs files). Radar is
deterministic, so this is a single point estimate.
"""
import json
import os
from collections import defaultdict

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
TEST_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer", "data", "test_set")
RADAR_OUTPUT = "/tmp/radar_repo/output.json"
OUT_DIR = os.path.join(BASE_DIR, "results_multisample", "solana_radar_baseline")

VULN_DISPLAY = {
    "missing_key_check": "Missing Key Check", "type_confusion": "Type Confusion",
    "cpi_reentrancy": "CPI Reentrancy", "unchecked_calls": "Unchecked External Calls",
    "integer_overflow": "Integer Overflow", "bump_seed": "Bump Seed",
    "dos": "Denial of Service", "not_vulnerable": "Not Vulnerable",
}
GROUND_TRUTH_MAP = {"integer_flow": "Integer Overflow", "cpi": "CPI Reentrancy"}

# Radar finding name -> this study's category. Findings with no mapping
# (Account Reinitialization, PDA Address Not Verified, Anchor Missing Min
# Output, Mint Configuration Unvalidated, Account Data Matching, PDA Sharing,
# Unvalidated Sysvar Account, ...) are excluded, same policy as the
# Tealer/Panda Algorand baselines: unmapped detectors are ignored rather than
# force-mapped, so they can never contribute a spurious FP here.
FINDING_TO_CATEGORY = {
    "Missing Signer Check": "Missing Key Check",
    "Missing Owner Check": "Missing Key Check",
    "Owner Check After CPI": "Missing Key Check",
    "Type Cosplay": "Type Confusion",
    "CPI Self Recursion": "CPI Reentrancy",
    "Missing Reentrancy Guards": "CPI Reentrancy",
    "Read-Only Reentrancy": "CPI Reentrancy",
    "Callback Token Reentrancy": "CPI Reentrancy",
    "Unchecked CPI Program Invoke": "Unchecked External Calls",
    "Unchecked Low-Level Call Return": "Unchecked External Calls",
    "Unvalidated CPI Program Account": "Unchecked External Calls",
    "Arbitrary Cross-Program Invocation": "Unchecked External Calls",
    "Unvalidated CPI Context Program": "Unchecked External Calls",
    "Integer Division Overflow": "Integer Overflow",
    "Unsafe Integer Downcast": "Integer Overflow",
    "Unchecked Arithmetics": "Integer Overflow",
    "Anchor Reward Overflow": "Integer Overflow",
    "Missing Bump Seed Canonicalization": "Bump Seed",
    "Invoke Signed Unvalidated Seeds": "Bump Seed",
    "Account Precreation DoS": "Denial of Service",
}


def load_ground_truth():
    gt = {}
    for f in sorted(os.listdir(TEST_DIR)):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(TEST_DIR, f), encoding="utf-8"))
        base = f.replace(".json", "")
        truth = d.get("vulnerability", "not_vulnerable")
        truth = GROUND_TRUTH_MAP.get(truth, truth)
        truth_display = VULN_DISPLAY.get(truth, truth)
        gt[base] = truth_display
    return gt


def load_predictions_per_file():
    findings = json.load(open(RADAR_OUTPUT, encoding="utf-8"))
    preds = defaultdict(set)
    parse_failed = set()
    for entry in findings:
        name = entry["name"]
        category = FINDING_TO_CATEGORY.get(name)
        if category is None:
            continue
        for loc in entry.get("locations", []):
            fname = os.path.basename(loc.split(":")[0])
            base = fname.replace(".rs", "")
            preds[base].add(category)
    return preds


PARSE_FAILED = {"solana_13", "solana_32", "solana_46"}  # tree-sitter "expected `!`"


def compute_metrics(results):
    tp = fp = tn = fn = 0
    per_vuln = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        truth = r["ground_truth_display"]
        preds = set(r["predictions"])
        if truth == "Not Vulnerable":
            if len(preds) == 0:
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
    gt = load_ground_truth()
    preds_per_file = load_predictions_per_file()

    results = []
    for base, truth_display in gt.items():
        if base in PARSE_FAILED:
            preds = []
            print(f"  [{base}] PARSE FAILED | truth={truth_display}")
        else:
            preds = sorted(preds_per_file.get(base, set()))
            print(f"  [{base}] truth={truth_display} | pred={preds}")
        results.append({"file": base, "ground_truth_display": truth_display, "predictions": preds})

    metrics = compute_metrics(results)
    n_errors = len(PARSE_FAILED)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "n_errors": n_errors, "details": results}, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_samples": 1, "note": "deterministic tool, single run", "n_errors": n_errors, "metrics": metrics}, f, indent=2)

    print(f"\nSUMMARY (Radar, Solana, 97-contract test set):")
    for k in ("accuracy", "precision", "recall", "f1_score"):
        print(f"  {k:10s}: {metrics[k]}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"  parse failures: {n_errors}/{len(results)}")


if __name__ == "__main__":
    main()
