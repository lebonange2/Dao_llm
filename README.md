# 🌿 DAO AI — Taoist-Aligned LLM Fine-Tuning

Fine-tune **Qwen2.5-7B-Instruct** with Taoist philosophy using **QLoRA** (4-bit quantization + LoRA adapters). Includes a multi-source data pipeline, a Gradio web UI for training on RunPod, and a local inference script.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start (RunPod)](#quick-start-runpod)
- [Quick Start (Local)](#quick-start-local)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Downloading the Model](#downloading-the-model)
- [Running the Model](#running-the-model)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Requirements](#requirements)

---

## Overview

DAO AI creates a Taoist-aligned language model by fine-tuning a general-purpose LLM on curated Taoist texts. The model learns to respond with the spirit of classical Taoist philosophy — wu wei, naturalness, and harmony.

**What it does:**
1. Scrapes and formats Taoist texts from multiple public-domain sources
2. Converts them into English instruction-response training pairs
3. Fine-tunes Qwen2.5-7B using QLoRA (runs on a single GPU with 16+ GB VRAM)
4. Produces a LoRA adapter you can run anywhere

---

## Project Structure

```
DAO AI/
├── main.py              # Full training pipeline (data → train → eval → merge)
├── app.py               # Gradio web UI for training on RunPod
├── sources.py           # Shared data source classes (used by main.py and scrape_data.py)
├── scrape_data.py       # Local data preparation script
├── run_model.py         # Local inference / interactive chat
├── start.sh             # RunPod entrypoint (installs deps, authenticates, launches)
├── Dockerfile           # Container image for RunPod deployment
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── expanded_pipeline.py # Reference: original multi-source pipeline design
```

---

## Quick Start (RunPod)

### 1. Deploy on RunPod

Create a GPU pod (A100 or A40 recommended) with the RunPod PyTorch template, then:

```bash
cd /workspace
git clone https://github.com/lebonange2/Dao_llm.git dao-llm
cd dao-llm
bash start.sh
```

### 2. Open the Gradio UI

The UI launches at `http://<pod-ip>:7860`. From there you can:
- Configure hyperparameters (epochs, batch size, LoRA rank, etc.)
- Upload a pre-built dataset or let it scrape on-the-fly
- Monitor training progress with live logs
- Download the trained model when finished

### 3. Set your HuggingFace token

Either set `HUGGINGFACE_TOKEN` as a RunPod environment variable, or enter it in the UI. Required because Qwen2.5-7B-Instruct is a gated model.

---

## Quick Start (Local)

### 1. Prepare the dataset on your machine

```bash
pip install requests beautifulsoup4

# Default: scrape ctext.org + sacred-texts.com
python scrape_data.py --preview

# English translations only (fastest, best quality)
python scrape_data.py --sources sacred_texts --preview

# All sources (needs HF_TOKEN for nanhuaijin)
python scrape_data.py --sources ctext sacred_texts nanhuaijin documentaries

# Offline mode (20 built-in Q&A pairs only)
python scrape_data.py --no-scrape
```

This produces `taoist_data.jsonl` — upload it in the Gradio UI or pass it directly:

```bash
python main.py --data_file taoist_data.jsonl
```

### 2. Train locally (needs GPU)

```bash
pip install -r requirements.txt
python main.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --data_file taoist_data.jsonl \
    --epochs 3 \
    --batch_size 2 \
    --lora_r 16
```

---

## Data Pipeline

### Sources

| Source | Key | Language | Content | License |
|---|---|---|---|---|
| Chinese Text Project | `ctext` | Classical Chinese | Dao De Jing, Zhuangzi, Liezi, Huainanzi | Public domain |
| Sacred Texts Archive | `sacred_texts` | English | Legge & Giles translations | Public domain |
| Nanhuaijin HF Dataset | `nanhuaijin` | Modern Chinese | Scholarly commentaries | Requires HF token |
| Documentary Metadata | `documentaries` | English | Film summaries | Fair use (metadata) |
| Built-in Seed Corpus | *(always included)* | English | 20 curated Q&A pairs | Original |

### Data Format

Each record in the JSONL file follows this structure:

```json
{
    "instruction": "What is wu wei?",
    "response": "Wu wei (無為) literally means 'non-doing' or 'effortless action.'...",
    "source": "fallback",
    "language": "english",
    "text": "### Instruction:\nWhat is wu wei?\n\n### Response:\nWu wei..."
}
```

### How it works

1. **`scrape_data.py`** (local) or **`main.py`** (RunPod fallback) instantiates source classes from `sources.py`
2. Each source scrapes its target, formats passages as instruction-response pairs
3. Built-in seed corpus is always appended as a floor
4. Deduplication removes repeated instructions
5. Output is saved as JSONL

---

## Training

### Pipeline Steps (main.py)

1. **Download model** — caches Qwen2.5-7B locally to avoid re-downloading
2. **Prepare dataset** — scrape sources or use uploaded file
3. **Format & tokenize** — applies chat template, tokenizes with truncation at 512 tokens
4. **Train** — QLoRA fine-tuning with HuggingFace `Trainer` + `DataCollatorForLanguageModeling`
5. **Evaluate** — generates responses to 5 test prompts
6. **Merge** — combines LoRA adapter with base model into a standalone model

### Default Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | Any HF causal LM works |
| Epochs | 3 | |
| Batch size | 2 | Per GPU |
| Gradient accumulation | 4 | Effective batch = 8 |
| Learning rate | 2e-4 | Cosine schedule with 3% warmup |
| LoRA rank | 16 | Higher = more capacity, more VRAM |
| LoRA alpha | 32 | Auto-set to 2× rank |
| Max sequence length | 512 | Truncates longer inputs |
| Quantization | 4-bit NF4 | Double quantization enabled |
| Optimizer | paged_adamw_8bit | Memory-efficient |

---

## Downloading the Model

After training completes in the UI:

1. **Scroll down** to the **📥 Download / Export Model** section (always visible below the tabs)
2. Choose **adapter** (~50 MB, fast) or **merged** (~14 GB, slower)
3. Click **📦 Prepare Zip**
4. Click the download link that appears

Alternatively, push directly to HuggingFace Hub from the same section.

### What's the difference?

| | Adapter | Merged |
|---|---|---|
| **Size** | ~50 MB | ~14 GB |
| **Contains** | LoRA weights only | Full model with LoRA baked in |
| **To run** | Needs base Qwen model + PEFT | Standalone, no base model needed |
| **Best for** | Sharing, iteration, storage | Deployment, serving |

---

## Running the Model

### Interactive chat

```bash
python run_model.py --adapter ./adapter
```

```
🧑 You: How should I deal with stress at work?
🌿 Dao: The Tao Te Ching teaches that the softest thing in the world
        overcomes the hardest...

🧑 You: quit
```

### Single prompt

```bash
python run_model.py --adapter ./adapter --prompt "What is the Tao?"
```

### From Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./adapter")
tokenizer = AutoTokenizer.from_pretrained("./adapter")

messages = [{"role": "user", "content": "What is wu wei?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Low VRAM (8 GB)

Add 4-bit quantization:

```python
from transformers import BitsAndBytesConfig

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./adapter")
```

---

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Required | Default | Description |
|---|---|---|---|
| `HUGGINGFACE_TOKEN` | Yes | — | HF token for gated model access |
| `BASE_MODEL` | No | `Qwen/Qwen2.5-7B-Instruct` | Base model ID |
| `OUTPUT_DIR` | No | `/workspace/taoist_finetuned` | Where to save trained model |
| `DATA_DIR` | No | `/workspace/taoist_data` | Where to save/load dataset |
| `MODEL_CACHE_DIR` | No | `/workspace/model_cache` | Local model download cache |
| `EPOCHS` | No | `3` | Training epochs |
| `BATCH_SIZE` | No | `2` | Per-GPU batch size |
| `GRAD_ACCUM` | No | `4` | Gradient accumulation steps |
| `LORA_R` | No | `16` | LoRA rank |
| `SKIP_SCRAPING` | No | `false` | Reuse existing dataset |
| `LAUNCH_MODE` | No | `ui` | `ui` for Gradio, `cli` for direct training |
| `GRADIO_PORT` | No | `7860` | UI port |

### CLI Arguments (main.py)

```
--base_model       HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)
--output_dir       Output directory (default: ./taoist_finetuned)
--data_dir         Data directory (default: ./taoist_data)
--data_file        Path to pre-built .jsonl file (skips scraping)
--sources          Data sources to scrape: ctext, sacred_texts, nanhuaijin, documentaries
--epochs           Training epochs (default: 3)
--batch_size       Per-GPU batch size (default: 2)
--grad_accum       Gradient accumulation steps (default: 4)
--lora_r           LoRA rank (default: 16)
--skip_scraping    Reuse existing dataset
--model_cache_dir  Local model cache (default: ./model_cache)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    scrape_data.py (local)                 │
│  sources.py → CText, SacredTexts, Nanhuaijin, Docs      │
│  Output: taoist_data.jsonl                               │
└────────────────────────┬─────────────────────────────────┘
                         │ upload via UI
┌────────────────────────▼─────────────────────────────────┐
│                     main.py (RunPod GPU)                  │
│                                                          │
│  1. Download base model (cached)                         │
│  2. Load/scrape dataset → tokenize                       │
│  3. QLoRA fine-tuning (Trainer + DataCollator)           │
│  4. Evaluate (5 Taoist alignment prompts)                │
│  5. Merge adapter + base → standalone model              │
│                                                          │
│  Output: adapter/ + merged_model/                        │
└────────────────────────┬─────────────────────────────────┘
                         │ download zip
┌────────────────────────▼─────────────────────────────────┐
│                  run_model.py (anywhere)                   │
│  Load base + adapter → interactive chat                  │
└──────────────────────────────────────────────────────────┘
```

---

## Requirements

- **Python** 3.10+
- **GPU** with 16+ GB VRAM (A100, A40, RTX 4090) — or 8 GB with 4-bit quantization
- **CUDA** 12.1+
- **HuggingFace account** with access to [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## License

Training data sourced from public-domain texts (pre-1928 translations by James Legge, Lionel Giles). The fine-tuned adapter weights are yours to use.

The base model (Qwen2.5-7B-Instruct) is subject to the [Qwen License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE).
