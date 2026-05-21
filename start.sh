#!/bin/bash
set -e

echo "========================================="
echo "  Taoist LLM Fine-Tuning — RunPod Setup  "
echo "========================================="

# ── Resolve script directory ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Install / verify Python dependencies ──
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "[INFO] Installing Python dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
    echo "[INFO] Dependencies installed."
fi

# ── HuggingFace login (required to download Qwen2.5-7B-Instruct) ──
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "[INFO] Logging in to HuggingFace..."
    # Try new CLI first, fall back to legacy
    if command -v hf &> /dev/null; then
        hf auth login --token "$HUGGINGFACE_TOKEN" 2>/dev/null || true
    elif command -v huggingface-cli &> /dev/null; then
        huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
    else
        python -c "from huggingface_hub import login; login(token='$HUGGINGFACE_TOKEN')" 2>/dev/null || true
    fi
    echo "[INFO] HuggingFace login complete."
else
    echo "[WARN] HUGGINGFACE_TOKEN not set. Download may fail for gated models."
fi

# ── Configurable hyperparameters via environment variables ──
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/taoist_finetuned}"
DATA_DIR="${DATA_DIR:-/workspace/taoist_data}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LORA_R="${LORA_R:-16}"
SKIP_SCRAPING="${SKIP_SCRAPING:-false}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/model_cache}"

echo "[INFO] Config:"
echo "  BASE_MODEL      = $BASE_MODEL"
echo "  MODEL_CACHE_DIR = $MODEL_CACHE_DIR"
echo "  OUTPUT_DIR      = $OUTPUT_DIR"
echo "  EPOCHS          = $EPOCHS"
echo "  BATCH_SIZE      = $BATCH_SIZE"
echo "  GRAD_ACCUM      = $GRAD_ACCUM"
echo "  LORA_R          = $LORA_R"

# Build the argument list
ARGS=(
    --base_model "$BASE_MODEL"
    --output_dir "$OUTPUT_DIR"
    --data_dir "$DATA_DIR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --grad_accum "$GRAD_ACCUM"
    --lora_r "$LORA_R"
    --model_cache_dir "$MODEL_CACHE_DIR"
)

if [ "$SKIP_SCRAPING" = "true" ]; then
    ARGS+=(--skip_scraping)
fi

# ── Launch mode: "ui" (default) or "cli" ──
LAUNCH_MODE="${LAUNCH_MODE:-ui}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

if [ "$LAUNCH_MODE" = "cli" ]; then
    echo "[INFO] CLI mode — starting training pipeline directly..."
    python /workspace/main.py "${ARGS[@]}"
    echo "[INFO] Done. Outputs saved to $OUTPUT_DIR"
else
    echo "[INFO] UI mode — launching Gradio web interface on port $GRADIO_PORT..."
    echo "[INFO] Open http://localhost:$GRADIO_PORT in your browser"
    echo "[INFO] On RunPod: expose port $GRADIO_PORT in the pod's HTTP port settings"
    GRADIO_PORT="$GRADIO_PORT" python /workspace/app.py
fi
