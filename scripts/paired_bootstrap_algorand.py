"""
Paired cluster bootstrap comparing two independently-evaluated model
configurations on the SAME fixed test set, for the TSE-reviewer-requested
significance check: Gemma-4-31B-FT+RAG vs. DeepSeek-V4-Pro+RAG, Algorand.

Resampling unit: contract index (0..44). Each bootstrap draw resamples
45 contract indices with replacement; for BOTH configurations we then pull
all 5 seeded observations for each resampled contract (same draw shared
across configs, preserving the pairing), aggregate TP/FP/TN/FN, and compute
F1 for each config at that draw. The distribution of (F1_A - F1_B) over
10,000 draws gives a 95% CI on the difference; if it excludes 0, the
difference is significant at alpha=0.05.
"""
import json
import os
import random
import statistics

RESULTS_ROOT = "/home/biasi/Documents/RAG4SCVulnDetection/results_multisample"
B = 10000
random.seed(20261101)

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
SAFE_LABELS_ALGORAND = {"Not Vulnerable", "No security risk", "No vulnerabilities"}


def classify_algorand(r):
    truth = MAP_TRUTH_TO_DISPLAY.get(r["ground_truth"], r["ground_truth"])
    preds = set(r["predictions"])
    preds_is_safe = len(preds) == 0 or preds.issubset(SAFE_LABELS_ALGORAND)
    if truth == "Not Vulnerable":
        if preds_is_safe:
            return (0, 0, 1, 0)
        real = preds - SAFE_LABELS_ALGORAND
        return (0, len(real), 0, 0)
    if truth in preds:
        extras = preds - {truth} - SAFE_LABELS_ALGORAND
        return (1, len(extras), 0, 0)
    wrong = preds - SAFE_LABELS_ALGORAND
    return (0, len(wrong), 0, 1)


def metrics_from_confusion(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_by_contract(dir_name, n_samples=5):
    """Returns {contract_file: [classify_tuple_per_seed, ...]} in fixed contract order."""
    path = os.path.join(RESULTS_ROOT, dir_name)
    by_contract = {}
    order = []
    for s in range(n_samples):
        with open(os.path.join(path, f"results_sample{s}.json")) as f:
            data = json.load(f)
        for r in data["details"]:
            fn = r["file"]
            if fn not in by_contract:
                by_contract[fn] = []
                order.append(fn)
            by_contract[fn].append(classify_algorand(r))
    return order, by_contract


def aggregate(tuples):
    tp = sum(t[0] for t in tuples)
    fp = sum(t[1] for t in tuples)
    tn = sum(t[2] for t in tuples)
    fn = sum(t[3] for t in tuples)
    return tp, fp, tn, fn


def paired_bootstrap(dir_a, dir_b, label_a, label_b, b=B):
    order_a, by_a = load_by_contract(dir_a)
    order_b, by_b = load_by_contract(dir_b)
    assert order_a == order_b, "Contract ordering mismatch between configs"
    n = len(order_a)

    f1_a_obs = metrics_from_confusion(*aggregate([t for c in order_a for t in by_a[c]]))[2]
    f1_b_obs = metrics_from_confusion(*aggregate([t for c in order_b for t in by_b[c]]))[2]

    diffs = []
    f1_as, f1_bs = [], []
    for _ in range(b):
        idx = [random.randrange(n) for _ in range(n)]
        tuples_a = [t for i in idx for t in by_a[order_a[i]]]
        tuples_b = [t for i in idx for t in by_b[order_b[i]]]
        f1_a = metrics_from_confusion(*aggregate(tuples_a))[2]
        f1_b = metrics_from_confusion(*aggregate(tuples_b))[2]
        f1_as.append(f1_a)
        f1_bs.append(f1_b)
        diffs.append(f1_a - f1_b)

    diffs.sort()
    lo = diffs[int(0.025 * b)]
    hi = diffs[int(0.975 * b) - 1]
    p_a_higher = sum(1 for d in diffs if d > 0) / b
    print(f"{label_a}: observed F1={f1_a_obs:.4f}")
    print(f"{label_b}: observed F1={f1_b_obs:.4f}")
    print(f"Observed diff (A-B) = {f1_a_obs - f1_b_obs:+.4f}")
    print(f"Paired cluster-bootstrap 95% CI on diff (A-B): [{lo:+.4f}, {hi:+.4f}]  (n_boot={b})")
    print(f"P(F1_A > F1_B) under resampling = {p_a_higher:.3f}")
    print(f"Significant at alpha=0.05 (CI excludes 0)? {'YES' if (lo > 0 or hi < 0) else 'NO'}")
    return {"f1_a": f1_a_obs, "f1_b": f1_b_obs, "diff_ci": (lo, hi), "p_a_higher": p_a_higher}


if __name__ == "__main__":
    print("=== Gemma-4-31B-FT+RAG vs. DeepSeek-V4-Pro+RAG, Algorand ===")
    paired_bootstrap(
        "algorand_gemma4_31b_ft_rag", "algorand_deepseek_rag",
        "Gemma-4-31B-FT+RAG", "DeepSeek-V4-Pro+RAG",
    )
    print()
    print("=== Qwen3-32B-FT+RAG vs. DeepSeek-V4-Pro+RAG, Algorand (no-RAG headline comparison context) ===")
    paired_bootstrap(
        "algorand_ft_rag", "algorand_deepseek_rag",
        "Qwen3-32B-FT+RAG", "DeepSeek-V4-Pro+RAG",
    )
