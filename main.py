#!/usr/bin/env python3
"""
End-to-End Taoist-Aligned LLM Fine-Tuning Script
Combines data scraping, instruction formatting, QLoRA fine-tuning, 
evaluation, and adapter merging into a single pipeline.

Dependencies:
  pip install torch transformers datasets accelerate trl peft bitsandbytes requests beautifulsoup4

Usage:
  python taoist_finetune.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./taoist_model \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --lora_r 16 \
    --skip_scraping  # Optional: use local data if already downloaded
"""

import os
import json
import re
import time
import logging
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Any

import torch

# ── Version guard: transformers >= 4.49 requires PyTorch >= 2.4 (DeviceMesh) ──
def _check_transformers_version():
    try:
        import importlib.metadata
        tf_ver = importlib.metadata.version("transformers")
        major, minor = (int(x) for x in tf_ver.split(".")[:2])
        if (major, minor) >= (4, 49):
            torch_major, torch_minor = (int(x) for x in torch.__version__.split(".")[:2])
            if (torch_major, torch_minor) < (2, 4):
                raise RuntimeError(
                    f"\n\n{'='*60}\n"
                    f"ERROR: transformers {tf_ver} requires PyTorch >= 2.4, "
                    f"but you have PyTorch {torch.__version__}.\n\n"
                    f"Fix: pip install 'transformers>=4.40.0,<4.49.0'\n"
                    f"  or: bash start.sh  (handles this automatically)\n"
                    f"{'='*60}\n"
                )
    except importlib.metadata.PackageNotFoundError:
        pass  # transformers not installed yet

_check_transformers_version()

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import snapshot_download

# ========================
# CONFIG & LOGGING
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

TAOIST_EVAL_PROMPTS = [
    "I'm stressed about meeting a deadline. What should I do?",
    "Is it better to act or not act?",
    "How can I become more successful than others?",
    "How should I handle conflict with a colleague?",
    "What does it mean to live in harmony with the Dao?"
]

# ========================
# MODEL DOWNLOAD
# ========================
def download_model_if_needed(model_id: str, cache_dir: str) -> str:
    """Download the base model to a local cache. Skip if already present."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Check if model files already exist in the cache
    model_marker = cache_path / "config.json"
    if model_marker.exists():
        logger.info(f"Model already cached at {cache_dir} — skipping download.")
        return str(cache_path)

    logger.info(f"Downloading model '{model_id}' to {cache_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(cache_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info(f"✅ Model downloaded to {cache_dir}")
    return str(cache_path)

# ========================
# DATA SOURCES
# ========================
from sources import (  # noqa: E402
    FALLBACK_PAIRS as _FALLBACK_PAIRS,
    SOURCE_MAP as _SOURCE_MAP,
)


def prepare_dataset(args: argparse.Namespace) -> Path:
    """Collect from all enabled sources and save as instruction-response JSONL."""
    output_path = Path(args.data_dir) / "taoist_corpus.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_scraping and output_path.exists():
        logger.info("Skipping scraping — loading existing dataset.")
        return output_path

    sources = getattr(args, "sources", ["ctext", "sacred_texts"])
    logger.info(f"Starting multi-source data collection: {sources}")

    all_pairs: List[Dict[str, Any]] = []
    for src_name in sources:
        cls = _SOURCE_MAP.get(src_name)
        if cls is None:
            logger.warning(f"Unknown source '{src_name}' — skipping.")
            continue
        src = cls()
        items = src.collect()
        all_pairs.extend(items)
        logger.info(f"✓ {src_name}: {len(items)} passages collected.")

    if not all_pairs:
        logger.warning("No data scraped — using built-in fallback corpus.")
        all_pairs = list(_FALLBACK_PAIRS)
    elif len(all_pairs) < 20:
        logger.warning("Only %d passages — padding with fallback corpus.", len(all_pairs))
        all_pairs = all_pairs + list(_FALLBACK_PAIRS)

    # Deduplicate
    seen: set = set()
    deduped = []
    for p in all_pairs:
        key = re.sub(r"\s+", "", p.get("instruction", p.get("text", "")))[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in deduped:
            record = {
                "instruction": pair.get("instruction", ""),
                "response":    pair.get("response", pair.get("text", "")),
                "source":      pair.get("source", "fallback"),
                "language":    pair.get("language", "english"),
                "text":        f"### Instruction:\n{pair.get('instruction','')}\n\n### Response:\n{pair.get('response','')}",
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    logger.info(f"✅ Saved {len(deduped)} passages to {output_path}")
    return output_path

# ========================
# DATASET FORMATTING
# ========================
def format_dataset(tokenizer: AutoTokenizer, data_path: Path, cache_dir: str = None) -> Dataset:
    """Convert passages into instruction-tuning format.

    Supports two JSONL formats:
      1. Rich format  — {"instruction": "...", "response": "..."}  (produced by scrape_data.py)
      2. Plain format — {"text": "..."}  (legacy / raw Chinese passages)
    """
    _fallback_instructions = [
        "How would a Daoist approach this teaching?",
        "Reflect on this in the spirit of wu wei:",
        "What does the Tao Te Ching suggest about this passage?",
        "Explain this from the perspective of natural harmony:",
    ]

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if "instruction" in item and "response" in item:
                user_msg = item["instruction"]
                asst_msg = item["response"]
            else:
                text = item.get("text", "")
                user_msg = _fallback_instructions[hash(text) % len(_fallback_instructions)]
                asst_msg = text

            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                formatted = f"User: {user_msg}\n\nAssistant: {asst_msg}"

            texts.append(formatted)

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    raw = Dataset.from_dict({"text": texts})
    dataset = raw.map(_tokenize, batched=True, remove_columns=["text"])
    logger.info(f"Tokenized {len(dataset)} instruction samples (columns: {dataset.column_names}).")
    return dataset

# ========================
# MODEL & TRAINING SETUP
# ========================
def setup_model_and_tokenizer(args: argparse.Namespace, local_model_path: str) -> tuple:
    """Load tokenizer, model, and apply QLoRA + PEFT."""
    logger.info(f"Loading tokenizer from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Configuring 4-bit quantization & LoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def run_training(model, tokenizer, dataset: Dataset, args: argparse.Namespace):
    """Execute SFT fine-tuning."""
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False
    )
    
    logger.info("Starting training...")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "adapter"))
    logger.info("✅ Training complete. Adapter saved.")

# ========================
# EVALUATION & MERGING
# ========================
def evaluate_model(model, tokenizer, args: argparse.Namespace):
    """Run quick Taoist alignment evaluation."""
    logger.info("Running Taoist alignment evaluation...")
    model.eval()
    for prompt in TAOIST_EVAL_PROMPTS:
        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = f"User: {prompt}\n\nAssistant:"
            
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7, do_sample=True, top_p=0.9
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nQ: {prompt}\nA: {response.split('Assistant:')[-1].strip()}\n" + "-"*60)

def merge_and_save(model, tokenizer, args: argparse.Namespace, local_model_path: str):
    """Merge LoRA adapter with base model."""
    logger.info("Merging adapter with base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base_model, os.path.join(args.output_dir, "adapter"))
    merged = merged.merge_and_unload()
    
    final_path = os.path.join(args.output_dir, "merged_model")
    merged.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"✅ Merged model saved to {final_path}")

# ========================
# MAIN PIPELINE
# ========================
def main():
    parser = argparse.ArgumentParser(description="End-to-End Taoist LLM Fine-Tuning")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./taoist_finetuned")
    parser.add_argument("--data_dir", type=str, default="./taoist_data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--skip_scraping", action="store_true")
    parser.add_argument("--sources", nargs="+",
                        default=["ctext", "sacred_texts"],
                        choices=list(_SOURCE_MAP.keys()),
                        help="Data sources to collect from (default: ctext sacred_texts)")
    parser.add_argument("--model_cache_dir", type=str, default="./model_cache",
                        help="Local directory to cache the downloaded base model")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Path to a pre-built .jsonl corpus file — skips all scraping")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 0. Download base model (skips if already cached)
        local_model_path = download_model_if_needed(
            args.base_model,
            os.path.join(args.model_cache_dir, args.base_model.replace('/', '_'))
        )
        
        # 1. Data
        if args.data_file and Path(args.data_file).exists():
            logger.info(f"Using uploaded data file: {args.data_file}")
            data_path = Path(args.data_file)
        else:
            if args.data_file:
                logger.warning(f"--data_file '{args.data_file}' not found — falling back to scraper.")
            data_path = prepare_dataset(args)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        dataset = format_dataset(tokenizer, data_path)
        
        # 2. Model
        model, tokenizer = setup_model_and_tokenizer(args, local_model_path)
        
        # 3. Train
        run_training(model, tokenizer, dataset, args)
        
        # 4. Evaluate
        evaluate_model(model, tokenizer, args)
        
        # 5. Merge
        merge_and_save(model, tokenizer, args, local_model_path)
        
        logger.info("🌿 Pipeline complete. May your model flow like water.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()