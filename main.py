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
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
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
# DATA COLLECTION
# ========================
def scrape_ctext_text(url: str, max_paragraphs: int = 500) -> List[str]:
    """Scrape classical Chinese text from ctext.org with respectful rate limiting."""
    headers = {"User-Agent": "TaoistLLM-Research/1.0 (Educational)"}
    texts = []
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for p in soup.select("div.ctext, p.chinese, .passage"):
            text = p.get_text(strip=True)
            if re.match(r"^[\u4e00-\u9fff\s，。？！；：、""''（）《》\-]+$", text) and len(text) > 15:
                texts.append(text)
                if len(texts) >= max_paragraphs:
                    break
        time.sleep(0.8)  # Respectful rate limit
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
    return texts

def prepare_dataset(args: argparse.Namespace) -> Path:
    """Download or load Taoist texts and save as JSONL."""
    output_path = Path(args.data_dir) / "taoist_corpus.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.skip_scraping and output_path.exists():
        logger.info("Skipping scraping. Loading existing dataset.")
        return output_path

    logger.info("Starting data collection...")
    urls = [
        "https://ctext.org/dao-de-jing",
        "https://ctext.org/zhuangzi",
        "https://ctext.org/liezi"
    ]
    
    corpus = []
    for url in urls:
        corpus.extend(scrape_ctext_text(url))
    
    if not corpus:
        logger.warning("No data scraped. Using minimal fallback corpus.")
        corpus = [
            "道可道，非常道；名可名，非常名。",
            "上善若水。水善利万物而不争，处众人之所恶，故几于道。",
            "天下皆知美之为美，斯恶已。皆知善之为善，斯不善已。"
        ]

    with open(output_path, "w", encoding="utf-8") as f:
        for text in corpus:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write("\n")
            
    logger.info(f"Saved {len(corpus)} passages to {output_path}")
    return output_path

# ========================
# DATASET FORMATTING
# ========================
def format_dataset(tokenizer: AutoTokenizer, data_path: Path, cache_dir: str = None) -> Dataset:
    """Convert raw passages into instruction-tuning format."""
    instructions = [
        "How would a Daoist approach this?",
        "Reflect on this teaching in the spirit of wu wei:",
        "What does the Dao De Jing suggest about this passage?",
        "Explain this from the perspective of natural harmony:"
    ]
    
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["text"]
            
            # Create instruction-response pair
            instruction = instructions[hash(text) % len(instructions)]
            
            # Use native chat template if available, else fallback
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": f"{instruction}\n{text}"},
                    {"role": "assistant", "content": f"Consider this teaching as water considers the earth: it does not force, yet shapes all things. {text}"}
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                formatted = f"User: {instruction}\n{text}\n\nAssistant: Consider this teaching as water considers the earth: it does not force, yet shapes all things. {text}"
                
            rows.append({"text": formatted})
            
    dataset = Dataset.from_list(rows)
    logger.info(f"Formatted {len(dataset)} instruction samples.")
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
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True
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
    parser.add_argument("--model_cache_dir", type=str, default="./model_cache",
                        help="Local directory to cache the downloaded base model")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 0. Download base model (skips if already cached)
        local_model_path = download_model_if_needed(
            args.base_model,
            os.path.join(args.model_cache_dir, args.base_model.replace('/', '_'))
        )
        
        # 1. Data
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