
🚀 How to Run with Expanded Sources
**************************************

# 1. Install dependencies
pip install torch transformers datasets accelerate trl peft bitsandbytes \
            requests beautifulsoup4 pandas youtube-transcript-api

# 2. Set environment variables (optional but recommended)
export HF_TOKEN=your_hf_token          # For Nanhuaijin dataset access
export COMMERCIAL_USE=false            # Critical for license compliance
export MERGE_ADAPTER=true              # Auto-merge LoRA adapter

# 3. Run with multiple sources
python taoist_finetune_expanded.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./taoist_model_v2 \
  --sources ctext sacred_texts nanhuaijin documentaries \
  --epochs 5 \
  --lora_r 32

# 4. Add academic abstracts (requires institutional access or manual collection)
# First, collect abstracts from Journal of Daoist Studies, Springer, MDPI [[14]][[13]]
# Save as academic_abstracts.jsonl with fields: {title, abstract, source, license}
# Then add to sources list: --sources ... academic_abstracts



📦 Post-Training: Evaluation & Alignment Checks
***************************************************
# eval_taoist_expanded.py - Quick alignment validation
EVAL_CATEGORIES = {
    "wu_wei": ["non-forcing", "effortless action", "spontaneity"],
    "paradox_tolerance": ["both/and", "complementary opposites", "yin-yang"],
    "nature_metaphors": ["water", "uncarved block", "valley spirit"],
    "anti_anthropocentrism": ["harmony with nature", "de-centering human", "cosmic perspective"]
}

def score_alignment(response: str, category: str) -> float:
    """Simple keyword-based alignment scoring (expand with semantic similarity)"""
    keywords = EVAL_CATEGORIES.get(category, [])
    response_lower = response.lower()
    return sum(1 for kw in keywords if kw in response_lower) / len(keywords)

# Run after training:
# for prompt in TAOIST_EVAL_PROMPTS:
#     response = generate(model, tokenizer, prompt)
#     for cat in EVAL_CATEGORIES:
#         print(f"{cat}: {score_alignment(response, cat):.2f}")


Quick Reference: Source Licensing Summary
*****************************************
Source,                       Training Allowed?,      Commercial Use?,        Attribution Required?
ctext.org classical texts,        ✅ Yes,                  ✅ Yes,                  ✅ Yes (ctext.org)
Sacred Texts (Legge/Giles),       ✅ Yes (pre-1928),       ✅ Yes,                  ✅ Yes (translator + archive)
Nanhuaijin HF dataset,            ✅ Research only,        ❌ No,                   ✅ Yes + publisher permission
Documentary metadata,             ✅ Fair use summary,     ⚠️ Case-by-case,         ✅ Yes (Films For Action)
Academic paper abstracts,         ✅ Usually,              ⚠️ Check publisher,      ✅ Yes (author + journal)
Modern commentaries (post-1970),  ⚠️ Verify per work,      ❌ Usually not,          ✅ Always
