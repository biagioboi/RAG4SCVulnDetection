"""
Decontamination check: token n-gram overlap between the public
algorand/pyteal GitHub repository (the plausible pre-training source
identified as a threat to validity in the paper) and the 45-contract
Algorand test set.

For each test contract, we compute the fraction of its 8-token n-grams
(on a simple identifier/keyword tokenization, case-sensitive) that appear
verbatim anywhere in the repo's .py source files, after stripping comments
and blank lines. A contract with a high max-overlap fraction against any
single repo file is flagged as a plausible contamination risk; this does
not prove the specific contract was seen during pre-training, only that
near-identical PyTeal code exists in a repository large frontier models
plausibly ingested.
"""
import json
import os
import re

REPO_DIR = "/tmp/pyteal_contamination_check"
TEST_DIR = "/home/biasi/Documents/RAG4SCVulnDetection/test_contracts"
N = 8  # n-gram size, in tokens


def tokenize(code):
    # Strip comments and blank lines, then split on identifier/operator boundaries.
    lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    text = " ".join(lines)
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_]", text)


def ngrams(tokens, n=N):
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def load_repo_ngrams():
    repo_ngrams = {}
    for root, _, files in os.walk(REPO_DIR):
        if "/tests/" in root or "/.git" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except Exception:
                continue
            toks = tokenize(code)
            if len(toks) >= N:
                repo_ngrams[path] = ngrams(toks)
    return repo_ngrams


def main():
    print("Cloning/loading repo n-grams...")
    repo_ngrams = load_repo_ngrams()
    print(f"Indexed {len(repo_ngrams)} .py files from algorand/pyteal "
          f"({sum(len(v) for v in repo_ngrams.values())} total {N}-grams)")

    test_files = sorted(f for f in os.listdir(TEST_DIR) if f.endswith(".json"))
    flagged = []
    overlaps = []
    for fn in test_files:
        with open(os.path.join(TEST_DIR, fn)) as f:
            data = json.load(f)
        code = data["smart_contract"]
        toks = tokenize(code)
        contract_ngrams = ngrams(toks)
        if not contract_ngrams:
            continue
        best_file, best_frac = None, 0.0
        for path, rng in repo_ngrams.items():
            overlap = len(contract_ngrams & rng)
            frac = overlap / len(contract_ngrams)
            if frac > best_frac:
                best_frac, best_file = frac, path
        overlaps.append(best_frac)
        print(f"{fn:16s} max_overlap={best_frac:.3f}  n_grams={len(contract_ngrams):4d}  best_match={os.path.relpath(best_file, REPO_DIR) if best_file else '-'}")
        if best_frac > 0.5:
            flagged.append((fn, best_frac, best_file))

    print(f"\n{len(flagged)}/{len(test_files)} contracts have >50% of their {N}-gram "
          f"vocabulary verbatim in algorand/pyteal:")
    for fn, frac, path in flagged:
        print(f"  {fn}: {frac:.3f} overlap with {os.path.relpath(path, REPO_DIR)}")

    print(f"\nOverlap distribution: min={min(overlaps):.3f} "
          f"median={sorted(overlaps)[len(overlaps)//2]:.3f} max={max(overlaps):.3f}")


if __name__ == "__main__":
    main()
