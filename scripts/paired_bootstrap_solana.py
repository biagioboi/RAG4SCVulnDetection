import json
import os
import random
import statistics

RESULTS_ROOT = "/home/biasi/Documents/RAG4SCVulnDetection/results_multisample"
B = 10000
random.seed(20261101)

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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_by_contract(dir_name, n_samples=5):
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
            by_contract[fn].append(classify_solana(r))
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
    assert order_a == order_b
    n = len(order_a)

    f1_a_obs = metrics_from_confusion(*aggregate([t for c in order_a for t in by_a[c]]))[2]
    f1_b_obs = metrics_from_confusion(*aggregate([t for c in order_b for t in by_b[c]]))[2]

    diffs = []
    for _ in range(b):
        idx = [random.randrange(n) for _ in range(n)]
        tuples_a = [t for i in idx for t in by_a[order_a[i]]]
        tuples_b = [t for i in idx for t in by_b[order_b[i]]]
        f1_a = metrics_from_confusion(*aggregate(tuples_a))[2]
        f1_b = metrics_from_confusion(*aggregate(tuples_b))[2]
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


if __name__ == "__main__":
    print("=== Gemma-4-31B FT-no-RAG vs. FT+RAG, Solana ===")
    paired_bootstrap(
        "solana_gemma4_31b_ft_norag", "solana_gemma4_31b_ft_rag",
        "Gemma-4-31B FT, no RAG", "Gemma-4-31B FT+RAG",
    )
