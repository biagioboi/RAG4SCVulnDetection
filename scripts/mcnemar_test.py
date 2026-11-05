"""
Exact McNemar's test (binomial form, appropriate for small discordant
counts) as a second, convergent significance test for the paper's two
headline SOTA comparisons, complementing the paired cluster bootstrap.

Per-contract binary correctness is defined by majority vote across the
5 stochastic seeds (correct if >=3/5 seeds correctly classify that
contract per this study's TP/TN definition), producing one paired binary
outcome per contract per model -- the natural unit for McNemar's test.
"""
import json
import os
from math import comb

RESULTS_ROOT = "/home/biasi/Documents/RAG4SCVulnDetection/results_multisample"

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


def is_correct(r):
    truth = MAP_TRUTH_TO_DISPLAY.get(r["ground_truth"], r["ground_truth"])
    preds = set(r["predictions"])
    preds_is_safe = len(preds) == 0 or preds.issubset(SAFE_LABELS_ALGORAND)
    if truth == "Not Vulnerable":
        return preds_is_safe
    return truth in preds


def per_contract_majority(dir_name, n_samples=5):
    path = os.path.join(RESULTS_ROOT, dir_name)
    votes = {}
    for s in range(n_samples):
        with open(os.path.join(path, f"results_sample{s}.json")) as f:
            data = json.load(f)
        for r in data["details"]:
            votes.setdefault(r["file"], []).append(is_correct(r))
    return {fn: (sum(v) >= (len(v) / 2)) for fn, v in votes.items()}


def exact_mcnemar(n01, n10):
    """Two-sided exact binomial McNemar test. Returns p-value."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return min(p, 1.0)


def compare(dir_a, dir_b, label_a, label_b):
    maj_a = per_contract_majority(dir_a)
    maj_b = per_contract_majority(dir_b)
    files = sorted(set(maj_a) & set(maj_b))
    n_both_correct = n_both_wrong = n01 = n10 = 0
    for fn in files:
        ca, cb = maj_a[fn], maj_b[fn]
        if ca and cb:
            n_both_correct += 1
        elif not ca and not cb:
            n_both_wrong += 1
        elif ca and not cb:
            n10 += 1  # A correct, B wrong
        else:
            n01 += 1  # A wrong, B correct
    p = exact_mcnemar(n01, n10)
    print(f"{label_a} vs {label_b} (n={len(files)} contracts, majority-vote over 5 seeds)")
    print(f"  Both correct: {n_both_correct}  Both wrong: {n_both_wrong}  "
          f"{label_a} only correct: {n10}  {label_b} only correct: {n01}")
    print(f"  Exact McNemar two-sided p-value: {p:.4f}  "
          f"({'significant' if p < 0.05 else 'NOT significant'} at alpha=0.05)")
    print()


if __name__ == "__main__":
    compare("algorand_gemma4_31b_ft_rag", "algorand_deepseek_rag",
            "Gemma-4-31B-FT+RAG", "DeepSeek-V4-Pro+RAG")
    compare("algorand_ft_rag", "algorand_deepseek_rag",
            "Qwen3-32B-FT+RAG", "DeepSeek-V4-Pro+RAG")
