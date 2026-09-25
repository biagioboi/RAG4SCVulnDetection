"""
Fine-Tuning Qwen3-32B for Solana Vulnerability Detection (QLoRA) -- v11.

Adds scripts/add_anchor_style_training_examples.py on top of v9's dataset:
5 more real Anchor-style pairs (Missing Key Check, DoS, CPI Reentrancy,
Unchecked External Calls, Integer Overflow) from coral-xyz/anchor's own
test/example programs, closing the code-style gap that made the newly
expanded test set's Anchor-style Missing Key Check cases (3/3) get missed --
until now every example the model had seen used native/raw processor.rs
style. Every one of the 7 vulnerable categories now has at least one
Anchor-style example and 15-19 vulnerable samples overall (was a mix of
16-19 with only Type Confusion/Bump Seed touched). 265 train / 25 validation
samples.
"""

import os
import json
import time

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"
REPO_DIR = os.path.join(BASE_DIR, "LLM-Contract-Analyzer")
DATA_DIR = os.path.join(REPO_DIR, "data", "training")

TRAIN_PATH = os.path.join(DATA_DIR, "dataset_think_format.jsonl")
VAL_PATH = os.path.join(DATA_DIR, "validation_dataset.jsonl")

WORK_DIR = os.path.join(REPO_DIR, "qwen3_training_v11")
os.makedirs(WORK_DIR, exist_ok=True)

MODEL_OUTPUT_DIR = os.path.join(REPO_DIR, "models", "Qwen3-32B-Solana-Audit-v11")
MAX_SEQ_LENGTH = 8192


def prepare_dataset(input_path, output_path):
    """Merge the separate thinking/content fields into a single CoT response."""
    prepared = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            new_messages = []
            for msg in sample["messages"]:
                if msg["role"] == "assistant" and "thinking" in msg:
                    merged = (
                        f"<think>\n{msg['thinking']}\n</think>\n\n"
                        f"<final>\n{msg['content']}\n</final>"
                    )
                    new_messages.append({"role": "assistant", "content": merged})
                elif msg["role"] == "developer":
                    new_messages.append({"role": "system", "content": msg["content"]})
                else:
                    new_messages.append(msg)
            prepared.append({"id": sample["id"], "messages": new_messages})

    with open(output_path, "w", encoding="utf-8") as f:
        for item in prepared:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(prepared)


def main():
    for path, name in [(TRAIN_PATH, "Training"), (VAL_PATH, "Validation")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} data not found at {path}")

    train_out = os.path.join(WORK_DIR, "train.jsonl")
    val_out = os.path.join(WORK_DIR, "val.jsonl")
    n_train = prepare_dataset(TRAIN_PATH, train_out)
    n_val = prepare_dataset(VAL_PATH, val_out)
    print(f"Prepared {n_train} training samples, {n_val} validation samples")

    import torch
    from unsloth import FastLanguageModel

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-32B",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    print("Base model loaded: Qwen3-32B (4-bit)")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    from datasets import load_dataset

    dataset_train = load_dataset("json", data_files=train_out, split="train")
    dataset_eval = load_dataset("json", data_files=val_out, split="train")
    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_eval)}")

    def convert_to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return {"text": text}

    dataset_train_fmt = dataset_train.map(convert_to_text, remove_columns=dataset_train.column_names)
    dataset_eval_fmt = dataset_eval.map(convert_to_text, remove_columns=dataset_eval.column_names)
    print("Sample preview:")
    print(dataset_train_fmt[0]["text"][:300])

    from trl import SFTConfig, SFTTrainer
    from unsloth import train_on_responses_only

    sft_config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=3,
        max_seq_length=MAX_SEQ_LENGTH,
        learning_rate=5e-5,
        warmup_steps=4,
        lr_scheduler_type="cosine",
        weight_decay=0.02,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        seed=3407,
        output_dir=os.path.join(WORK_DIR, "checkpoints"),
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train_fmt,
        eval_dataset=dataset_eval_fmt,
        args=sft_config,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Starting training...")
    start = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - start
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print(f"Final training loss: {trainer_stats.metrics.get('train_loss')}")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Adapter saved to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
