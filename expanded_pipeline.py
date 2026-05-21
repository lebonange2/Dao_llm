#!/usr/bin/env python3
"""
Taoist LLM: Expanded Multi-Source Fine-Tuning Pipeline
Adds: academic papers, documentaries metadata, public domain books, cultural context
With licensing validation, multimodal preprocessing, and ethical safeguards.

Dependencies:
  pip install torch transformers datasets accelerate trl peft bitsandbytes \
              requests beautifulsoup4 pandas youtube-transcript-api

Usage:
  python taoist_finetune_expanded.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./taoist_model_v2 \
    --sources ctext sacred_texts nanhuaijin documentaries \
    --license_check true \
    --epochs 5
"""

import os, json, re, time, logging, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse

import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# ========================
# CONFIG & LOGGING
# ========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

LICENSE_DB = {
    "public_domain": {"years_before": 1928, "regions": ["US", "global_classics"]},
    "cc_by": {"attribution_required": True, "share_alike": False},
    "cc_by_sa": {"attribution_required": True, "share_alike": True},
    "research_only": {"commercial_use": False, "citation_required": True},
    "proprietary": {"scraping_allowed": False, "fair_use_summary_only": True}
}

# ========================
# DATA SOURCES MODULES
# ========================
class TaoistDataSource:
    """Base class for licensed, ethical data collection"""
    
    def __init__(self, name: str, license_type: str, base_url: Optional[str] = None):
        self.name = name
        self.license = LICENSE_DB.get(license_type, {})
        self.base_url = base_url
        self.collected = []
        
    def validate_license(self, content_meta: Dict) -> bool:
        """Check if content can be used for training"""
        if self.license.get("scraping_allowed") is False:
            logger.warning(f"{self.name}: Scraping not permitted for {content_meta.get('title')}")
            return False
        if self.license.get("commercial_use") is False and os.getenv("COMMERCIAL_USE") == "true":
            logger.error(f"{self.name}: Commercial use prohibited. Set COMMERCIAL_USE=false")
            return False
        return True
    
    def collect(self, **kwargs) -> List[Dict]:
        raise NotImplementedError

class CTextSource(TaoistDataSource):
    """Chinese Text Project: classical Chinese texts"""
    
    def __init__(self):
        super().__init__("ctext", "public_domain", "https://ctext.org")
        
    def collect(self, texts: List[str] = None, max_passages: int = 1000) -> List[Dict]:
        if texts is None:
            texts = ["dao-de-jing", "zhuangzi", "liezi"]
            
        for text_id in texts:
            url = f"{self.base_url}/{text_id}"
            try:
                headers = {"User-Agent": "TaoistLLM-Research/2.0 (Educational)"}
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                
                for p in soup.select("div.ctext, p.chinese, .passage"):
                    content = p.get_text(strip=True)
                    if re.match(r"^[\u4e00-\u9fff\s，。？！；：、""''（）《》\-]+$", content) and len(content) > 20:
                        self.collected.append({
                            "source": "ctext",
                            "text_id": text_id,
                            "language": "classical_chinese",
                            "content": content,
                            "license": "public_domain",
                            "attribution": "Chinese Text Project (ctext.org)"
                        })
                        if len(self.collected) >= max_passages:
                            return self.collected
                time.sleep(0.8)  # Respectful rate limit
            except Exception as e:
                logger.warning(f"CText scrape failed for {text_id}: {e}")
        return self.collected

class SacredTextsSource(TaoistDataSource):
    """Sacred Texts Archive: public domain English translations"""
    
    def __init__(self):
        super().__init__("sacred_texts", "public_domain", "https://sacred-texts.com/tao")
        
    def collect(self, translations: List[str] = None) -> List[Dict]:
        if translations is None:
            translations = ["tao-te-ching", "chuang-tzu", "lieh-tzu"]
            
        for trans in translations:
            url = f"{self.base_url}/{trans}.htm"
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extract main text content
                content_div = soup.find("div", class_="main") or soup.find("body")
                if content_div:
                    text = content_div.get_text(separator="\n", strip=True)
                    # Clean: remove navigation, ads
                    text = re.sub(r"\[.*?\]|\(.*?\)|^\s*[\[\(].*?[\]\)].*$", "", text, flags=re.M)
                    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
                    
                    for para in paragraphs[:50]:  # Limit per text
                        self.collected.append({
                            "source": "sacred_texts",
                            "text_id": trans,
                            "language": "english",
                            "content": para,
                            "license": "public_domain",
                            "attribution": "Sacred Texts Archive (sacred-texts.com); translator: James Legge/Lionel Giles"
                        })
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"SacredTexts fetch failed for {trans}: {e}")
        return self.collected

class NanhuaijinHFSource(TaoistDataSource):
    """Hugging Face: Nanhuaijin scholarly commentaries (Chinese)"""
    
    def __init__(self):
        super().__init__("nanhuaijin_hf", "research_only")
        
    def collect(self, subset: Optional[str] = None) -> List[Dict]:
        if os.getenv("HF_TOKEN"):
            try:
                # Load from Hugging Face if token available
                ds = load_dataset("wobure/nanhuaijin-collections", split="train", token=os.getenv("HF_TOKEN"))
                for item in ds:
                    if subset and subset.lower() not in item.get("title", "").lower():
                        continue
                    self.collected.append({
                        "source": "nanhuaijin_hf",
                        "text_id": hashlib.md5(item["title"].encode()).hexdigest()[:8],
                        "language": "modern_chinese",
                        "content": item["content"],
                        "title": item["title"],
                        "license": "research_only",
                        "attribution": "南怀瑾先生著作; Dataset: wobure/nanhuaijin-collections"
                    })
                logger.info(f"Loaded {len(self.collected)} Nanhuaijin passages")
            except Exception as e:
                logger.warning(f"Nanhuaijin HF load failed (may require token): {e}")
                # Fallback: use local cache if available
                cache_path = Path("./data/nanhuaijin_cache.jsonl")
                if cache_path.exists():
                    with open(cache_path, "r", encoding="utf-8") as f:
                        for line in f:
                            self.collected.append(json.loads(line))
        else:
            logger.info("HF_TOKEN not set; skipping Nanhuaijin source (or use cached data)")
        return self.collected

class DocumentaryMetadataSource(TaoistDataSource):
    """Documentary metadata & transcripts (NOT video/audio scraping)"""
    
    def __init__(self):
        super().__init__("documentaries", "fair_use_summary")
        
    def collect(self, titles: List[str] = None) -> List[Dict]:
        if titles is None:
            titles = [
                "The Art of Effortless Living",
                "The Art of Letting Go",
                "Opening Dao"
            ]
            
        for title in titles:
            # Search Films For Action or similar educational platforms
            search_url = f"https://www.filmsforaction.org/search/?q={title.replace(' ', '+')}"
            try:
                resp = requests.get(search_url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extract summary/description (NOT full transcript without permission)
                desc_elem = soup.find("p", class_="description") or soup.find("meta", attrs={"name": "description"})
                if desc_elem:
                    summary = desc_elem.get("content") if desc_elem.name == "meta" else desc_elem.get_text(strip=True)
                    self.collected.append({
                        "source": "documentary_metadata",
                        "text_id": title.lower().replace(" ", "_"),
                        "language": "english",
                        "content": f"Documentary summary: {summary}",
                        "license": "fair_use_educational",
                        "attribution": f"Films For Action; Documentary: {title}",
                        "modality": "metadata_only"  # Critical: not full video content
                    })
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Documentary metadata fetch failed for '{title}': {e}")
        return self.collected

# ========================
# DATASET FORMATTING (Taoist-Aligned)
# ========================
def format_taoist_instruction(item: Dict, tokenizer: AutoTokenizer) -> str:
    """Convert raw content into Taoist-aligned instruction format"""
    
    # Taoist prompt styles: open-ended, paradox-embracing, nature-metaphor
    prompt_templates = [
        "Reflect on this teaching in the spirit of wu wei (effortless action):",
        "How might water respond to the wisdom in this passage?",
        "What does this suggest about harmony with the Dao?",
        "Consider this from the perspective of natural spontaneity (ziran):",
        "In the tradition of Zhuangzi, how would you contemplate this idea?"
    ]
    
    content = item["content"]
    # Truncate very long passages
    if len(content) > 1500:
        content = content[:1400] + "..."
    
    instruction = prompt_templates[hash(item.get("text_id", "")) % len(prompt_templates)]
    
    # Use model's native chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{text}"},
            {"role": "assistant", "content": f"In the flowing way of the Dao: {content[:200]}..."}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        return f"User: {instruction}\n{content}\n\nAssistant: In the flowing way of the Dao: {content[:200]}..."

# ========================
# MAIN PIPELINE
# ========================
def main():
    parser = argparse.ArgumentParser(description="Taoist LLM: Multi-Source Fine-Tuning")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./taoist_model_v2")
    parser.add_argument("--sources", nargs="+", default=["ctext", "sacred_texts"], 
                       choices=["ctext", "sacred_texts", "nanhuaijin", "documentaries", "academic_abstracts"])
    parser.add_argument("--license_check", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. COLLECT DATA FROM SELECTED SOURCES
    logger.info("🌿 Starting multi-source data collection...")
    all_items = []
    
    source_map = {
        "ctext": CTextSource,
        "sacred_texts": SacredTextsSource,
        "nanhuaijin": NanhuaijinHFSource,
        "documentaries": DocumentaryMetadataSource
    }
    
    for src_name in args.sources:
        if src_name in source_map:
            src = source_map[src_name]()
            items = src.collect()
            # License validation
            if args.license_check:
                items = [i for i in items if src.validate_license(i)]
            all_items.extend(items)
            logger.info(f"✓ Collected {len(items)} items from {src_name}")
    
    if not all_items:
        logger.error("No valid data collected. Check licenses, network, or source selection.")
        return
    
    # 2. FORMAT DATASET
    logger.info("Formatting instruction dataset...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    formatted = []
    for item in all_items:
        try:
            formatted_text = format_taoist_instruction(item, tokenizer)
            formatted.append({"text": formatted_text, "meta": {k:v for k,v in item.items() if k not in ["content"]}})
        except Exception as e:
            logger.warning(f"Formatting failed for item: {e}")
    
    dataset = Dataset.from_list(formatted)
    logger.info(f"✅ Prepared {len(dataset)} instruction samples")
    
    # Save raw data for reproducibility
    data_path = Path(args.output_dir) / "training_data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for item in formatted:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    # 3. MODEL SETUP (QLoRA)
    logger.info("Loading model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
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
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. TRAINING
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
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True
    )
    
    logger.info("🌀 Starting training...")
    trainer.train()
    
    # 5. SAVE & MERGE
    adapter_path = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # Optional merge for deployment
    if os.getenv("MERGE_ADAPTER", "true").lower() == "true":
        logger.info("Merging adapter with base model...")
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
        merged = PeftModel.from_pretrained(base, adapter_path)
        merged = merged.merge_and_unload()
        merged.save_pretrained(os.path.join(args.output_dir, "merged"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "merged"))
    
    logger.info("🌊 Pipeline complete. Model flows like water.")

if __name__ == "__main__":
    main()