"""
Major Concern 2 response: CoT-isolation ablation.

Trains one of three additional configurations, using the identical base
model, LoRA recipe, optimizer, schedule, and seed as the paper's headline
Qwen3-32B-pyteal-desc run (extracted verbatim from finetuning_unsloth.ipynb),
varying only the training data and the assistant-response template:

  Config A (--config A): same 500 examples as the full multi-task recipe
      (Task 1 + Task 2 + Task 3), but the assistant target is collapsed to
      the <final> verdict only -- no <think> reasoning trace is ever
      supervised, for any example. Isolates the marginal value of CoT
      supervision while holding task breadth and example count fixed.
  Config B (--config B): Task 1 only (200 examples, generic vulnerability
      detection, full <think>+<final> as in the main recipe). Isolates how
      much of the full recipe's effect the core CoT-reasoning task alone
      achieves, without the Task 2/3 auxiliary supervision.
  Config C (--config C): Task 2 only (200 examples, targeted vulnerability
      verification, full <think>+<final>). Isolates the auxiliary
      hypothesis-verification task alone.

Config D (the full multi-task recipe) is not retrained here -- it is the
already-trained models/Qwen3-32B-pyteal-desc/checkpoint-189.

Usage: python scripts/train_qwen3_32b_ablation.py --config {A,B,C}
"""
import argparse
import os

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only

BASE_DIR = "/home/biasi/Documents/RAG4SCVulnDetection"

CONFIGS = {
    "A": {
        "train_file": "datasets/dataset_think_format.jsonl",
        "eval_file": "datasets/validation_dataset.jsonl",
        "model_id": "Qwen3-32B-ablation-A-final-only",
        "template_mode": "final_only",
    },
    "B": {
        "train_file": "datasets/dataset_configB_task1_cot.jsonl",
        "eval_file": "datasets/validation_configB_task1.jsonl",
        "model_id": "Qwen3-32B-ablation-B-task1-cot",
        "template_mode": "full",
    },
    "C": {
        "train_file": "datasets/dataset_configC_task2_targeted.jsonl",
        "eval_file": "datasets/validation_configC_task2.jsonl",
        "model_id": "Qwen3-32B-ablation-C-task2-targeted",
        "template_mode": "full",
    },
}

TEMPLATE_FULL = """
{% for message in messages %}
{% if message['role'] == 'developer' or message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>\n

{% elif message['role'] == 'user' %}
<|im_start|>user{{ message['content'] }}<|im_end|>user\n
{% elif message['role'] == 'assistant' %}
{% if message['description'] %}
<|im_start|>assistant
<final>
{{ message['description'] }}
</final>
<|im_end|>\n
{% else %}
<|im_start|>assistant
<think>
{{ message['thinking'] }}</think>\n
<final>
{{ message['content'] }}
</final><|im_end|>\n
{% endif %}
{% endif %}
{% endfor %}

{%- if add_generation_prompt %}
{%- if think %}
<|im_start|>assistant
<think>
{%- else %}
<|im_start|>assistant
<final>
{%- endif %}
{%- endif %}
"""

TEMPLATE_FINAL_ONLY = """
{% for message in messages %}
{% if message['role'] == 'developer' or message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>\n

{% elif message['role'] == 'user' %}
<|im_start|>user{{ message['content'] }}<|im_end|>user\n
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
<final>
{% if message['description'] %}{{ message['description'] }}{% else %}{{ message['content'] }}{% endif %}
</final><|im_end|>\n
{% endif %}
{% endfor %}

{%- if add_generation_prompt %}
<|im_start|>assistant
<final>
{%- endif %}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "C"], required=True)
    args = parser.parse_args()
    cfg = CONFIGS[args.config]

    max_seq_length = 2048
    model_name = "unsloth/Qwen3-32B"
    new_model_id = cfg["model_id"]

    print(f"=== Training Config {args.config}: {new_model_id} ===")
    print(f"Train file: {cfg['train_file']} | Template: {cfg['template_mode']}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer.chat_template = TEMPLATE_FULL if cfg["template_mode"] == "full" else TEMPLATE_FINAL_ONLY

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset_train = load_dataset("json", data_files=os.path.join(BASE_DIR, cfg["train_file"]), split="train").shuffle(seed=43)
    dataset_eval = load_dataset("json", data_files=os.path.join(BASE_DIR, cfg["eval_file"]), split="train").shuffle(seed=43)

    dataset_train = standardize_sharegpt(dataset_train)
    dataset_train = dataset_train.map(formatting_prompts_func, batched=True)
    dataset_eval = standardize_sharegpt(dataset_eval)
    dataset_eval = dataset_eval.map(formatting_prompts_func, batched=True)

    print(f"Train examples: {len(dataset_train)} | Eval examples: {len(dataset_eval)}")
    print("--- sample rendered training text ---")
    print(dataset_train[0]["text"][:800])

    sft_config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=3,
        max_seq_length=max_seq_length,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        weight_decay=0.02,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=5,
        seed=3407,
        output_dir=os.path.join(BASE_DIR, "models", new_model_id),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        args=sft_config,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user",
        response_part="<|im_end|>user",
    )

    print("--- sample labels (masked) ---")
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " ")[:500])

    trainer_stats = trainer.train()
    print(trainer_stats)
    print(f"=== Config {args.config} training complete. Saved to models/{new_model_id} ===")


if __name__ == "__main__":
    main()
