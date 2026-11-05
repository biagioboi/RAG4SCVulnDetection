"""
Recompute headline Algorand F1 excluding the 4 contracts flagged by
scripts/contamination_check.py (>50% 8-gram overlap with algorand/pyteal):
pyteal5, pyteal20, pyteal22, pyteal23.
"""
import json
import os

RESULTS_ROOT = "/home/biasi/Documents/RAG4SCVulnDetection/results_multisample"
FLAGGED = {"pyteal5.json", "pyteal20.json", "pyteal22.json", "pyteal23.json"}

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


def classify(r):
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


def metrics(tp, fp, tn, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return p, r, f1, acc


def f1_for_config(dir_name, n_samples=5, exclude=None):
    path = os.path.join(RESULTS_ROOT, dir_name)
    tp = fp = tn = fn = 0
    for s in range(n_samples):
        fpath = os.path.join(path, f"results_sample{s}.json")
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        for r in data["details"]:
            if exclude and r["file"] in exclude:
                continue
            t = classify(r)
            tp += t[0]; fp += t[1]; tn += t[2]; fn += t[3]
    return metrics(tp, fp, tn, fn)


CONFIGS = [
    ("algorand_qwen3_base_norag", "Qwen3-32B base, no RAG"),
    ("algorand_qwen3_base_rag", "Qwen3-32B base + RAG"),
    ("algorand_qwen3_ft_norag", "Qwen3-32B FT, no RAG"),
    ("algorand_ft_rag", "Qwen3-32B FT + RAG"),
    ("algorand_qwq_base_norag", "QwQ-32B base, no RAG"),
    ("algorand_qwq_base_rag", "QwQ-32B base + RAG"),
    ("algorand_qwq_ft_norag", "QwQ-32B FT, no RAG"),
    ("algorand_qwq_ft_rag", "QwQ-32B FT + RAG"),
    ("algorand_deepseek_norag", "DeepSeek-V4-Pro, no RAG"),
    ("algorand_deepseek_rag", "DeepSeek-V4-Pro + RAG"),
    ("algorand_gemma4_31b_norag", "Gemma-4-31B base, no RAG"),
    ("algorand_gemma4_31b_rag", "Gemma-4-31B base + RAG"),
    ("algorand_gemma4_31b_ft_norag", "Gemma-4-31B FT, no RAG"),
    ("algorand_gemma4_31b_ft_rag", "Gemma-4-31B FT + RAG"),
]

print(f"{'Config':32s} {'F1 (all 45)':>12s} {'F1 (41, decontam)':>18s} {'delta':>8s}")
for dir_name, label in CONFIGS:
    _, _, f1_all, _ = f1_for_config(dir_name)
    _, _, f1_clean, _ = f1_for_config(dir_name, exclude=FLAGGED)
    print(f"{label:32s} {f1_all:12.4f} {f1_clean:18.4f} {f1_clean - f1_all:+8.4f}")
