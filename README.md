# RAG4SCVulnDetection

Chain-of-Thought fine-tuning and Retrieval-Augmented Generation for LLM-based vulnerability
detection in low-resource smart-contract languages: **Algorand/PyTeal** and **Solana/Rust**.

This repository contains the full pipeline (dataset construction, QLoRA fine-tuning, RAG,
evaluation harness, and the LaTeX source of the accompanying paper) for both languages under a
single project, including the former `LLM-Contract-Analyzer` codebase (Solana), now merged in
under `LLM-Contract-Analyzer/`.

## Pretrained models (Hugging Face)

LoRA adapters, fine-tuned on top of `unsloth/Qwen3-32B` and `unsloth/QwQ-32B`:

| Model | Language | Link |
|---|---|---|
| Qwen3-32B (FT) | Algorand/PyTeal | [biagioboi/qwen3-32b-pyteal-vuln](https://huggingface.co/biagioboi/qwen3-32b-pyteal-vuln) |
| QwQ-32B (FT) | Algorand/PyTeal | [biagioboi/qwq-32b-pyteal-vuln](https://huggingface.co/biagioboi/qwq-32b-pyteal-vuln) |
| Qwen3-32B (FT) | Solana/Rust | [biagioboi/qwen3-32b-solana-vuln](https://huggingface.co/biagioboi/qwen3-32b-solana-vuln) |
| QwQ-32B (FT) | Solana/Rust | [biagioboi/qwq-32b-solana-vuln](https://huggingface.co/biagioboi/qwq-32b-solana-vuln) |

## Datasets (Hugging Face)

Chain-of-Thought (`<think>`/`<final>`) instruction-tuning corpora used to produce the models above:

| Dataset | Language | Link |
|---|---|---|
| PyTeal vulnerability CoT | Algorand | [biagioboi/pyteal-vuln-cot](https://huggingface.co/datasets/biagioboi/pyteal-vuln-cot) |
| Solana vulnerability CoT | Solana | [biagioboi/solana-vuln-cot](https://huggingface.co/datasets/biagioboi/solana-vuln-cot) |

Local copies of the same JSONL files are also tracked in this repo (`datasets/` for Algorand,
`LLM-Contract-Analyzer/data/training/` for Solana) for direct use by the training/evaluation
scripts without a network dependency.

## Repository layout

- `scripts/` -- fine-tuning and multi-sample evaluation scripts for Algorand.
- `datasets/` -- Algorand Chain-of-Thought training/validation data.
- `results_multisample/` -- 5-seed mean/std evaluation results for every model x RAG configuration, both languages.
- `paper/` -- LaTeX source of the paper (`main.tex`).
- `images/` -- matplotlib-generated figures used in the paper.
- `LLM-Contract-Analyzer/` -- Solana pipeline (dataset construction, training, RAG, evaluation), merged into this repo.

Model weights and training checkpoints are intentionally excluded from git (see `.gitignore`)
and are distributed via Hugging Face instead (links above).
