"""
Parser for extracting vulnerability verdicts from model responses.
Adapted from Nicola Tortora's parser.py for Solana/Rust vulnerabilities.
"""

import re

# Solana/Rust vulnerability names (all recognized variants)
VULNERABILITIES = [
    "Missing Key Check",
    "Access Control",
    "Type Confusion",
    "Input Validation",
    "CPI Reentrancy",
    "Reentrancy",
    "Unchecked External Calls",
    "Unchecked Calls",
    "Integer Overflow",
    "Integer Underflow",
    "Integer Flow",
    "Bump Seed",
    "Bump Seed Canonicalization",
    "Denial of Service",
    "DoS",
    "Not Vulnerable",
    "No security risk",
    "No vulnerabilities",
    "None detected",
]


def parse_response(response: str) -> tuple:
    """Parse model response to extract vulnerability findings.

    Returns:
        (success, vulnerabilities): success is True if a verdict was
        recognized and at least one known vulnerability was found in it.
    """
    tag_pattern = re.compile(
        r"<final>\s*(.*?)\s*</final>", re.IGNORECASE | re.DOTALL
    )
    match = tag_pattern.search(response)

    if match:
        content = match.group(1).strip()
    else:
        # Fallback: some models (especially non-fine-tuned base models under
        # longer RAG prompts) skip the <final> wrapper and write the verdict
        # directly as "### Vulnerability: ...". Without this fallback such
        # responses were silently scored as "no vulnerability found",
        # inflating true negatives and destroying recall/precision.
        think_end = re.search(r"</think>", response, re.IGNORECASE)
        content = response[think_end.end():] if think_end else response
        content = content.strip()

    # Only read the explicit "### Vulnerability: ..." verdict line(s), not
    # the whole <final> block: the Explanation/Risk prose routinely mentions
    # other category names in passing (e.g. "...preventing reentrancy") which
    # previously got matched as if they were separate findings.
    vuln_lines = re.findall(r"###\s*Vulnerability:\s*(.+)", content, re.IGNORECASE)

    found_vulns = []
    for line in vuln_lines:
        for part in re.split(r",|;|\band\b", line, flags=re.IGNORECASE):
            part = part.strip().lower()
            if not part:
                continue
            for v in VULNERABILITIES:
                if v.lower() == part or v.lower() in part:
                    if v not in found_vulns:
                        found_vulns.append(v)
                    break

    if len(found_vulns) > 0:
        return True, found_vulns
    else:
        return False, []
