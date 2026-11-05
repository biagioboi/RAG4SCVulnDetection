"""
Cluster bootstrap confidence intervals for every headline F1/precision/recall
number reported in the paper, computed purely from already-saved per-contract
raw predictions (results_multisample/<config>/results_sampleN.json) -- no new
model inference required.

Resampling unit: one (seed, contract) observation. For each bootstrap
iteration we draw, with replacement, as many observations as exist in the
pooled sample (5 seeds x |test set|), re-aggregate the per-contract
TP/FP/TN/FN classification (identical rule used by every eval script in this
repo, reproduced below for Algorand and Solana), and recompute
precision/recall/F1/accuracy from the resampled confusion matrix. This is a
standard cluster/case bootstrap: contracts (and seeds) are the natural
resampling unit, not individual TP/FP increments, since several increments
can originate from the same contract (a model can report >1 spurious
vulnerability on one contract).
"""
import json
import os
import random
import statistics

RESULTS_ROOT = "/home/biasi/Documents/RAG4SCVulnDetection/results_multisample"
B = 10000
random.seed(20261007)

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

VULNERABILITY_ALIASES_SOLANA = {
    "Missing Key Check": ["missing key check", "access control"],
    "Type Confusion": ["type confusion", "input validation"],
    "CPI Reentrancy": ["cpi reentrancy", "cpi", "reentrancy"],
    "Unchecked External Calls": ["unchecked external calls", "unchecked calls"],
    "Integer Overflow": ["integer overflow", "integer underflow", "integer flow"],
    "Bump Seed": ["bump seed", "bump seed canonicalization"],
    "Denial of Service": ["denial of service", "dos"],
    "Not Vulnerable": ["not vulnerable", "no security risk", "no vulnerabilities", "none detected"],
}
GROUND_TRUTH_MAP_SOLANA = {"integer_flow": "Integer Overflow", "cpi": "CPI Reentrancy"}


def normalize_predictions_solana(predictions):
    canonical_set = set()
    for pred in predictions:
        pred_lower = pred.lower().strip()
        for canonical, aliases in VULNERABILITY_ALIASES_SOLANA.items():
            if pred_lower in aliases:
                canonical_set.add(canonical)
                break
    return canonical_set


def classify_algorand(r):
    """Returns a (tp, fp, tn, fn) tuple for one contract observation."""
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


def classify_solana(r):
    truth = r["ground_truth"]
    predictions = normalize_predictions_solana(r["predictions"])
    if truth in GROUND_TRUTH_MAP_SOLANA:
        truth = GROUND_TRUTH_MAP_SOLANA[truth]
    if truth == "Not Vulnerable":
        if predictions == {"Not Vulnerable"} or len(predictions) == 0:
            return (0, 0, 1, 0)
        real_preds = predictions - {"Not Vulnerable"}
        return (0, len(real_preds), 0, 0)
    if truth in predictions:
        extras = predictions - {truth, "Not Vulnerable"}
        return (1, len(extras), 0, 0)
    wrong = predictions - {"Not Vulnerable"}
    return (0, len(wrong), 0, 1)


def metrics_from_confusion(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return precision, recall, f1, accuracy


def load_observations(config_dir, language, n_samples):
    classify = classify_algorand if language == "algorand" else classify_solana
    obs = []
    for s in range(n_samples):
        path = os.path.join(config_dir, f"results_sample{s}.json")
        with open(path) as f:
            data = json.load(f)
        for r in data["details"]:
            obs.append(classify(r))
    return obs


def bootstrap_ci(obs, b=B):
    n = len(obs)
    f1s, precs, recs, accs = [], [], [], []
    for _ in range(b):
        sample = [obs[random.randrange(n)] for _ in range(n)]
        tp = sum(o[0] for o in sample)
        fp = sum(o[1] for o in sample)
        tn = sum(o[2] for o in sample)
        fn = sum(o[3] for o in sample)
        p, r, f1, acc = metrics_from_confusion(tp, fp, tn, fn)
        precs.append(p); recs.append(r); f1s.append(f1); accs.append(acc)
    def ci(vals):
        vals = sorted(vals)
        lo = vals[int(0.025 * b)]
        hi = vals[int(0.975 * b) - 1]
        return statistics.mean(vals), lo, hi
    return {"precision": ci(precs), "recall": ci(recs), "f1": ci(f1s), "accuracy": ci(accs)}


CONFIGS = [
    # (dir_name, language, n_samples, display_name)
    ("algorand_qwen3_base_norag", "algorand", 5, "Qwen3-32B base, no RAG"),
    ("algorand_qwen3_base_rag", "algorand", 5, "Qwen3-32B base + RAG"),
    ("algorand_qwen3_ft_norag", "algorand", 5, "Qwen3-32B FT, no RAG"),
    ("algorand_ft_rag", "algorand", 5, "Qwen3-32B FT + RAG"),
    ("algorand_qwq_base_norag", "algorand", 5, "QwQ-32B base, no RAG"),
    ("algorand_qwq_base_rag", "algorand", 5, "QwQ-32B base + RAG"),
    ("algorand_qwq_ft_norag", "algorand", 5, "QwQ-32B FT, no RAG"),
    ("algorand_qwq_ft_rag", "algorand", 5, "QwQ-32B FT + RAG"),
    ("algorand_deepseek_norag", "algorand", 5, "DeepSeek-V4-Pro, no RAG"),
    ("algorand_deepseek_rag", "algorand", 5, "DeepSeek-V4-Pro + RAG"),
    ("solana_deepseek_norag", "solana", 5, "DeepSeek-V4-Pro, no RAG"),
    ("solana_deepseek_rag", "solana", 5, "DeepSeek-V4-Pro + RAG"),
    ("solana_base_norag", "solana", 5, "Qwen3-32B base, no RAG"),
    ("solana_base_rag", "solana", 5, "Qwen3-32B base + RAG"),
    ("solana_ft_norag", "solana", 5, "Qwen3-32B FT, no RAG"),
    ("solana_ft_rag_k2", "solana", 5, "Qwen3-32B FT + RAG (k=2)"),
    ("algorand_gemma4_31b_norag", "algorand", 5, "Gemma-4-31B base, no RAG"),
    ("algorand_gemma4_31b_rag", "algorand", 5, "Gemma-4-31B base + RAG"),
    ("algorand_gemma4_31b_ft_norag", "algorand", 5, "Gemma-4-31B FT, no RAG"),
    ("algorand_gemma4_31b_ft_rag", "algorand", 5, "Gemma-4-31B FT + RAG"),
    ("solana_gemma4_31b_norag", "solana", 5, "Gemma-4-31B base, no RAG"),
    ("solana_gemma4_31b_rag", "solana", 5, "Gemma-4-31B base + RAG"),
    ("solana_gemma4_31b_ft_norag", "solana", 5, "Gemma-4-31B FT, no RAG"),
    ("solana_gemma4_31b_ft_rag", "solana", 5, "Gemma-4-31B FT + RAG"),
    ("algorand_qwen3coder_norag", "algorand", 5, "Qwen3-Coder-30B base, no RAG"),
    ("algorand_qwen3coder_rag", "algorand", 5, "Qwen3-Coder-30B base + RAG"),
    ("algorand_qwen3coder_ft_norag", "algorand", 5, "Qwen3-Coder-30B FT, no RAG"),
    ("algorand_qwen3coder_ft_rag", "algorand", 5, "Qwen3-Coder-30B FT + RAG"),
    ("solana_qwen3coder_norag", "solana", 5, "Qwen3-Coder-30B base, no RAG"),
    ("solana_qwen3coder_rag", "solana", 5, "Qwen3-Coder-30B base + RAG"),
    ("solana_qwen3coder_ft_norag", "solana", 5, "Qwen3-Coder-30B FT, no RAG"),
]

# the last config (solana FT+RAG) is handled separately below, using
# however many samples are actually present on disk right now.


def main():
    results = {}
    for dir_name, language, n_samples, display in CONFIGS:
        path = os.path.join(RESULTS_ROOT, dir_name)
        n_available = sum(
            1 for s in range(n_samples)
            if os.path.exists(os.path.join(path, f"results_sample{s}.json"))
        )
        obs = load_observations(path, language, n_available)
        ci = bootstrap_ci(obs)
        results[dir_name] = {"display": display, "language": language, "n_samples": n_available, "ci": ci}
        f1_mean, f1_lo, f1_hi = ci["f1"]
        print(f"{language:8s} {display:32s} n_obs={len(obs):4d} (n_samples={n_available}) "
              f"F1={f1_mean:.3f} 95% CI=[{f1_lo:.3f}, {f1_hi:.3f}]")

    # Solana Qwen3-Coder FT+RAG: pick up however many samples exist now.
    dir_name = "solana_qwen3coder_ft_rag"
    path = os.path.join(RESULTS_ROOT, dir_name)
    n_available = sum(
        1 for s in range(5) if os.path.exists(os.path.join(path, f"results_sample{s}.json"))
    )
    if n_available > 0:
        obs = load_observations(path, "solana", n_available)
        ci = bootstrap_ci(obs)
        results[dir_name] = {"display": "Qwen3-Coder-30B FT + RAG", "language": "solana",
                              "n_samples": n_available, "ci": ci}
        f1_mean, f1_lo, f1_hi = ci["f1"]
        print(f"solana   {'Qwen3-Coder-30B FT + RAG':32s} n_obs={len(obs):4d} (n_samples={n_available}) "
              f"F1={f1_mean:.3f} 95% CI=[{f1_lo:.3f}, {f1_hi:.3f}]  <-- PARTIAL, pipeline still running")

    with open("/tmp/bootstrap_ci_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full results to /tmp/bootstrap_ci_results.json")


if __name__ == "__main__":
    main()
