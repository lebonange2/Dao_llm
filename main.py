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
# DATA SOURCES
# ========================

_PROMPT_TEMPLATES = [
    "Reflect on this teaching in the spirit of wu wei (effortless action):",
    "How might water respond to the wisdom in this passage?",
    "What does this suggest about harmony with the Dao?",
    "Consider this from the perspective of natural spontaneity (ziran):",
    "In the tradition of Zhuangzi, how would you contemplate this idea?",
    "What practical wisdom does this Taoist passage offer?",
    "Explain this teaching from the Tao Te Ching.",
]

_FALLBACK_PAIRS = [
    {"instruction": "What is the Tao?",
     "response": "The Tao (道) is the fundamental principle underlying all of reality. It cannot be fully captured in words — 'The Tao that can be told is not the eternal Tao.' It is the source from which all things arise, the nameless beginning of heaven and earth. Rather than a deity or a force to be commanded, it is the natural way of the universe itself: silent, effortless, and inexhaustible."},
    {"instruction": "What is wu wei?",
     "response": "Wu wei (無為) literally means 'non-doing' or 'effortless action.' It does not mean laziness or passivity, but acting in complete harmony with the natural flow of things — without force, without ego, without unnecessary interference. The sage accomplishes everything through wu wei: 'Do nothing, and nothing is left undone.'"},
    {"instruction": "What does the Tao Te Ching teach about water?",
     "response": "Water is one of the central metaphors in the Tao Te Ching. 'The highest good is like water' (上善若水). Water benefits all living things without competing, dwelling willingly in the lowest places that others disdain. Nothing in the world is softer than water, yet nothing surpasses it in overcoming the hard and strong."},
    {"instruction": "What does Taoism say about the relationship between opposites?",
     "response": "Taoism sees opposites not as contradictions but as complementary aspects of a single whole. Beauty only exists because we recognise ugliness; good only because we know bad. This interdependence is captured in the image of yin and yang: two forces that generate and balance each other in endless, dynamic harmony."},
    {"instruction": "What is the Taoist meaning of simplicity?",
     "response": "Simplicity (樸, pu — the uncarved block) is one of Taoism's core values. The uncarved block represents potential in its purest form, before it has been shaped by ambition, social roles, or cultural conditioning. 'Manifest plainness, embrace simplicity, reduce selfishness, have few desires.'"},
    {"instruction": "How should a leader govern according to Taoist philosophy?",
     "response": "The Tao Te Ching describes the ideal ruler as one who governs through stillness and restraint. 'Governing a large state is like cooking a small fish' — handle it too much and you ruin it. The best leader is barely noticed by the people; when the work is done the people say: 'We did it ourselves.'"},
    {"instruction": "What does Taoism teach about the acceptance of death?",
     "response": "Taoism views death not as a tragedy but as a natural transformation. When Zhuangzi's wife died, he was found singing. Clinging to life and fearing death is like a child refusing to go home at the end of the day. The sage accepts the endless transformations of existence without resistance."},
    {"instruction": "What does the Tao Te Ching say about the power of emptiness?",
     "response": "Thirty spokes converge at the hub of a wheel, but it is the empty space at the centre that makes the wheel useful. Clay is shaped into a vessel, but it is the hollow space inside that makes it useful. Therefore, what exists serves for profit, but what does not exist serves for utility."},
    {"instruction": "What is the Taoist view of knowledge and wisdom?",
     "response": "'In pursuit of learning, every day something is added. In pursuit of the Tao, every day something is dropped.' True wisdom is not about accumulating facts but about shedding the ego and the compulsion to control. Knowing others is intelligence; knowing yourself is true wisdom."},
    {"instruction": "How does Taoism approach conflict and competition?",
     "response": "The Tao Te Ching states: 'The way of heaven benefits without harming; the way of the sage acts without contending.' Like water, the Taoist does not fight for high places but naturally finds the level. The soft overcomes the hard; the gentle overcomes the rigid."},
    {"instruction": "What can we learn from the Tao about living with uncertainty?",
     "response": "The Tao teaches that uncertainty and change are the nature of reality. 'Fortune and misfortune take turns with each other.' The sage does not grasp tightly at any outcome. By remaining flexible, present, and empty of fixed expectations, one can respond appropriately to whatever arises."},
    {"instruction": "What is the significance of the three treasures in Taoism?",
     "response": "The Tao Te Ching describes three treasures: compassion (慈), frugality (儉), and not daring to be first in the world (不敢為天下先). Compassion leads to true courage; frugality creates abundance; not presuming to lead allows one to become a true leader when the time comes."},
    {"instruction": "How does Taoist philosophy relate to nature?",
     "response": "Nature is the primary teacher in Taoism. The Tao expresses itself through the patterns of mountains, rivers, seasons, and living things. Human beings are not separate from nature but part of its flow. The Taoist sage acts as nature acts: purposefully but without forcing, consistently but without rigidity."},
    {"instruction": "What is the Taoist teaching on self-knowledge?",
     "response": "'Knowing others is wisdom; knowing yourself is enlightenment. Mastering others requires force; mastering yourself requires strength.' The inward journey is the foundation of Taoist practice. When one knows oneself deeply, action becomes effortless and flows from genuine nature."},
    {"instruction": "How does the Zhuangzi describe the ideal Taoist sage?",
     "response": "In the Zhuangzi, the ideal sage moves through the world like wind through trees — present, responsive, and leaving no trace. The famous story of Cook Ding illustrates this: his knife never dulls because he follows the natural structure of the animal, not forcing his way. This is Taoist mastery: moving effortlessly along the grain of reality."},
    {"instruction": "How can Taoist philosophy be applied to modern daily life?",
     "response": "Wu wei suggests: before acting, pause and ask whether action is truly necessary. In work, allow projects to develop at their natural pace. In relationships, practice listening and yielding. Simplify — reduce unnecessary possessions and mental noise. Spend time in nature. Notice the rhythms in your life and work with them rather than against them."},
    {"instruction": "Explain the Taoist concept of yin and yang.",
     "response": "Yin and yang are complementary aspects of a single reality. Yin is associated with receptivity, darkness, softness, the moon, and winter. Yang with activity, light, hardness, the sun, and summer. Each contains the seed of the other. Health and wisdom come from honouring both rather than insisting on one."},
    {"instruction": "What does the Tao Te Ching say about happiness and contentment?",
     "response": "'He who knows that enough is enough will always have enough.' Chasing wealth, fame, or pleasure is like trying to fill a leaking vessel. True contentment arises when one aligns with the Tao and stops measuring life against an imagined ideal. The sage finds the extraordinary within the ordinary."},
    {"instruction": "What is the Taoist teaching on flexibility and resilience?",
     "response": "Taoism prizes flexibility as the mark of life and rigidity as the mark of death. 'A man is born gentle and supple. At his death he is hard and stiff.' The tree that survives the storm is the one that bends. Resilience comes not from hardening against difficulty but from remaining fluid enough to flow around it."},
    {"instruction": "What does the Tao Te Ching say about war and violence?",
     "response": "'Weapons are instruments of fear; they are not a wise man's tools.' The military leader who delights in conquest is unfit to lead. Even in unavoidable conflict, the Taoist approach is to use minimum force and mourn those who fall on both sides. The greatest victories are won without battle."},
]


class TaoistDataSource:
    """Base class for licensed, ethical Taoist data collection."""

    def __init__(self, name: str, base_url: str = None):
        self.name = name
        self.base_url = base_url
        self.collected: List[Dict[str, Any]] = []

    def collect(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


class CTextSource(TaoistDataSource):
    """Chinese Text Project — classical Chinese texts (public domain)."""

    TEXTS = ["dao-de-jing", "zhuangzi", "liezi", "huainanzi"]

    def __init__(self):
        super().__init__("ctext", "https://ctext.org")

    def collect(self, texts: List[str] = None, max_passages: int = 500) -> List[Dict[str, Any]]:
        texts = texts or self.TEXTS
        headers = {"User-Agent": "TaoistLLM-Research/2.0 (Educational)"}
        for text_id in texts:
            url = f"{self.base_url}/{text_id}"
            try:
                logger.info(f"  [ctext] Fetching {text_id} …")
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for p in soup.select("div.ctext, p.chinese, .passage"):
                    content = p.get_text(strip=True)
                    if re.match(r"^[\u4e00-\u9fff\s，。？！；：、""''（）《》\-]+$", content) and len(content) > 20:
                        prompt = _PROMPT_TEMPLATES[hash(content) % len(_PROMPT_TEMPLATES)]
                        self.collected.append({
                            "source": "ctext", "language": "classical_chinese",
                            "instruction": f"{prompt}\n\n{content}",
                            "response": f"This classical passage embodies the Taoist way — effortless, natural, and pointing beyond words to the living flow of the Tao. {content}",
                        })
                        if len(self.collected) >= max_passages:
                            return self.collected
                time.sleep(0.8)
            except Exception as e:
                logger.warning(f"ctext scrape failed for {text_id}: {e}")
        return self.collected


class SacredTextsSource(TaoistDataSource):
    """Sacred Texts Archive — public domain English translations (Legge, Giles)."""

    TRANSLATIONS = {
        "tao/taote":    "Tao Te Ching (Legge)",
        "tao/chuang":   "Zhuangzi (Legge)",
        "tao/lieh":     "Liezi (Giles)",
    }

    def __init__(self):
        super().__init__("sacred_texts", "https://sacred-texts.com")

    def collect(self, max_per_text: int = 80) -> List[Dict[str, Any]]:
        for path, title in self.TRANSLATIONS.items():
            url = f"{self.base_url}/{path}.htm"
            try:
                logger.info(f"  [sacred_texts] Fetching {title} …")
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                body = soup.find("div", class_="main") or soup.body
                if not body:
                    continue
                raw = body.get_text(separator="\n", strip=True)
                raw = re.sub(r"\[.*?\]|\(c\).*", "", raw)
                paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 60]
                count = 0
                for para in paragraphs:
                    if count >= max_per_text:
                        break
                    prompt = _PROMPT_TEMPLATES[hash(para) % len(_PROMPT_TEMPLATES)]
                    self.collected.append({
                        "source": "sacred_texts", "language": "english",
                        "instruction": f"{prompt}\n\n\"{para}\"",
                        "response": (
                            f"This passage from {title} reflects a core Taoist principle. "
                            "It invites us to align with the natural flow of the Tao — acting without forcing, "
                            "knowing without grasping, and finding strength in yielding. "
                            "In practice, this means embodying wu wei: effortless action arising from "
                            "one's deepest, most natural state of being."
                        ),
                    })
                    count += 1
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"sacred_texts fetch failed for {path}: {e}")
        return self.collected


class NanhuaijinHFSource(TaoistDataSource):
    """Hugging Face — Nanhuaijin scholarly commentaries (requires HF_TOKEN)."""

    def __init__(self):
        super().__init__("nanhuaijin_hf")

    def collect(self, **kwargs) -> List[Dict[str, Any]]:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logger.info("  [nanhuaijin] HF_TOKEN not set — skipping.")
            return []
        try:
            from datasets import load_dataset as _load_dataset
            logger.info("  [nanhuaijin] Loading from HuggingFace …")
            ds = _load_dataset("wobure/nanhuaijin-collections", split="train", token=token)
            for item in ds:
                content = item.get("content", "")[:1400]
                if len(content) < 30:
                    continue
                prompt = _PROMPT_TEMPLATES[hash(content) % len(_PROMPT_TEMPLATES)]
                self.collected.append({
                    "source": "nanhuaijin_hf", "language": "modern_chinese",
                    "instruction": f"{prompt}\n\n{content}",
                    "response": f"In the spirit of Master Nan's commentaries: {content[:300]}…",
                })
            logger.info(f"  [nanhuaijin] Loaded {len(self.collected)} passages.")
        except Exception as e:
            logger.warning(f"nanhuaijin load failed: {e}")
        return self.collected


class DocumentaryMetadataSource(TaoistDataSource):
    """Films For Action — documentary summaries (metadata only, not full transcripts)."""

    TITLES = ["The Art of Effortless Living", "The Art of Letting Go", "Opening Dao"]

    def __init__(self):
        super().__init__("documentaries", "https://www.filmsforaction.org")

    def collect(self, **kwargs) -> List[Dict[str, Any]]:
        for title in self.TITLES:
            url = f"{self.base_url}/search/?q={title.replace(' ', '+')}"
            try:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.text, "html.parser")
                meta = soup.find("meta", attrs={"name": "description"})
                desc = soup.find("p", class_="description")
                summary = (meta.get("content") if meta else None) or (desc.get_text(strip=True) if desc else None)
                if summary and len(summary) > 40:
                    self.collected.append({
                        "source": "documentaries", "language": "english",
                        "instruction": f"What is the Taoist teaching explored in the documentary '{title}'?",
                        "response": f"The documentary '{title}' explores: {summary}",
                    })
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Documentary metadata failed for '{title}': {e}")
        return self.collected


_SOURCE_MAP: Dict[str, type] = {
    "ctext":         CTextSource,
    "sacred_texts":  SacredTextsSource,
    "nanhuaijin":    NanhuaijinHFSource,
    "documentaries": DocumentaryMetadataSource,
}


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

    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if "instruction" in item and "response" in item:
                # ── Rich format: use instruction + response directly ──
                user_msg = item["instruction"]
                asst_msg = item["response"]
            else:
                # ── Plain format: wrap raw text in a generic prompt ──
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
        max_seq_length=512,
        tokenizer=tokenizer,
        packing=False
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