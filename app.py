#!/usr/bin/env python3
"""
Gradio Web UI for Taoist LLM Fine-Tuning Pipeline.
Accessible at http://0.0.0.0:7860 — forward this port on RunPod.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

import gradio as gr

SCRIPT_PATH = Path(__file__).parent / "main.py"

# Global handle for the running subprocess
_training_process: subprocess.Popen | None = None
_training_active: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _zip_directory(source_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in source_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(source_dir.parent))
    return zip_path


def _build_args(base_model, output_dir, data_dir, epochs,
                batch_size, grad_accum, lora_r, skip_scraping) -> list[str]:
    args = [
        sys.executable, str(SCRIPT_PATH),
        "--base_model", base_model,
        "--output_dir", output_dir,
        "--data_dir",   data_dir,
        "--epochs",     str(int(epochs)),
        "--batch_size", str(int(batch_size)),
        "--grad_accum", str(int(grad_accum)),
        "--lora_r",     str(int(lora_r)),
    ]
    if skip_scraping:
        args.append("--skip_scraping")
    return args


# ── Training (streaming generator) ───────────────────────────────────────────

def start_and_stream(
    base_model, output_dir, data_dir,
    epochs, batch_size, grad_accum, lora_r,
    skip_scraping, hf_token,
):
    global _training_process, _training_active

    if _training_active:
        yield (
            "⚠️  Training is already running — wait for it to finish or stop it first.\n",
            gr.update(value="⚠️ Already running", visible=True),
            gr.update(interactive=False),
        )
        return

    env = os.environ.copy()
    if hf_token.strip():
        env["HUGGINGFACE_TOKEN"] = hf_token.strip()
        env["HF_TOKEN"]          = hf_token.strip()

    args = _build_args(base_model, output_dir, data_dir,
                       epochs, batch_size, grad_accum, lora_r, skip_scraping)

    header = (
        "🌿  Taoist LLM Fine-Tuning Pipeline\n"
        + "─" * 60 + "\n"
        + "Command: " + " ".join(args) + "\n"
        + "─" * 60 + "\n"
    )
    accumulated = header

    _training_active = True
    yield accumulated, gr.update(value="🔄 Training in progress…", visible=True), gr.update(interactive=False)

    try:
        _training_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in _training_process.stdout:
            accumulated += line
            yield accumulated, gr.update(), gr.update(interactive=False)

        _training_process.wait()
        rc = _training_process.returncode

    except Exception as exc:
        accumulated += f"\n❌  Unexpected error: {exc}\n"
        _training_active = False
        yield accumulated, gr.update(value="❌ Error", visible=True), gr.update(interactive=True)
        return

    _training_active = False

    if rc == 0:
        accumulated += f"\n{'─'*60}\n✅  Training complete!  Model saved to: {output_dir}\n"
        yield accumulated, gr.update(value="✅ Training complete!", visible=True), gr.update(interactive=True)
    else:
        accumulated += f"\n{'─'*60}\n❌  Training failed (exit code {rc})\n"
        yield accumulated, gr.update(value=f"❌ Failed (exit code {rc})", visible=True), gr.update(interactive=True)


def stop_training():
    global _training_process, _training_active
    if _training_process and _training_active:
        _training_process.terminate()
        _training_active = False
        return "⛔  Training stopped by user."
    return "No active training to stop."


# ── Download / Export ─────────────────────────────────────────────────────────

def prepare_download(output_dir: str):
    out = Path(output_dir)
    merged  = out / "merged_model"
    adapter = out / "adapter"

    target = merged if merged.exists() else (adapter if adapter.exists() else None)
    if target is None:
        return None, "❌  No model found at that path. Run training first."

    zip_path = out / f"{target.name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    _zip_directory(target, zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    label = "merged_model" if target == merged else "LoRA adapter"
    return str(zip_path), f"✅  {label} zipped → {zip_path.name}  ({size_mb:.1f} MB)"


def push_to_hub(output_dir: str, repo_id: str, hub_token: str):
    if not repo_id.strip():
        return "❌  Enter a HuggingFace repo ID (e.g. username/taoist-qwen-7b)."
    if not hub_token.strip():
        return "❌  HuggingFace token is required to push."

    out     = Path(output_dir)
    merged  = out / "merged_model"
    adapter = out / "adapter"
    target  = merged if merged.exists() else (adapter if adapter.exists() else None)

    if target is None:
        return "❌  No model found. Run training first."

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hub_token.strip())
        api.create_repo(repo_id=repo_id.strip(), exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(target),
            repo_id=repo_id.strip(),
            commit_message="Upload Taoist fine-tuned model via DAO AI UI",
        )
        return f"✅  Pushed to https://huggingface.co/{repo_id.strip()}"
    except Exception as exc:
        return f"❌  Push failed: {exc}"


# ── Build UI ──────────────────────────────────────────────────────────────────

_CSS = """
.log-box textarea { font-family: 'Courier New', monospace !important; font-size: 12px !important; }
.status-ok  { color: #16a34a !important; font-weight: 600; }
.status-err { color: #dc2626 !important; font-weight: 600; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    title="🌿 Taoist LLM Fine-Tuning",
    css=_CSS,
) as demo:

    gr.Markdown("""
# 🌿 Taoist LLM Fine-Tuning UI
Fine-tune **Qwen2.5-7B-Instruct** with Taoist philosophy using **QLoRA** on RunPod GPU instances.
""")

    with gr.Tabs():

        # ── Tab 1 : Configure & Train ─────────────────────────
        with gr.Tab("⚙️  Configure & Train"):

            with gr.Row():
                with gr.Column():
                    base_model_inp = gr.Textbox(
                        value="Qwen/Qwen2.5-7B-Instruct",
                        label="Base Model (HuggingFace ID)",
                    )
                    hf_token_inp = gr.Textbox(
                        label="HuggingFace Token",
                        placeholder="hf_…  (required for gated models)",
                        type="password",
                    )
                with gr.Column():
                    output_dir_inp = gr.Textbox(value="/workspace/taoist_finetuned", label="Output Directory")
                    data_dir_inp   = gr.Textbox(value="/workspace/taoist_data",      label="Data Directory")

            with gr.Row():
                epochs_inp     = gr.Slider(1, 10, value=3,  step=1,  label="Epochs")
                batch_size_inp = gr.Slider(1, 8,  value=2,  step=1,  label="Batch Size (per GPU)")
                grad_accum_inp = gr.Slider(1, 16, value=4,  step=1,  label="Gradient Accumulation")
                lora_r_inp     = gr.Slider(4, 64, value=16, step=4,  label="LoRA Rank (r)")

            skip_scraping_inp = gr.Checkbox(
                label="Skip web scraping — reuse already-downloaded corpus",
                value=False,
            )

            with gr.Row():
                train_btn = gr.Button("🚀  Start Training", variant="primary", scale=4)
                stop_btn  = gr.Button("⛔  Stop",           variant="stop",    scale=1)

            status_md = gr.Textbox(label="Status", interactive=False, visible=False)

        # ── Tab 2 : Live Training Logs ────────────────────────
        with gr.Tab("📋  Training Logs"):
            log_box = gr.Textbox(
                label="Live Output",
                lines=35,
                max_lines=35,
                interactive=False,
                show_copy_button=True,
                elem_classes=["log-box"],
                placeholder="Logs will stream here once training starts…",
            )

        # ── Tab 3 : Download / Export ─────────────────────────
        with gr.Tab("📥  Download / Export"):

            gr.Markdown("### 📦  Download Model Weights")
            gr.Markdown(
                "Zips the **merged model** (if available) or the **LoRA adapter** "
                "from the output directory and serves it for download."
            )
            with gr.Row():
                dl_dir_inp = gr.Textbox(value="/workspace/taoist_finetuned", label="Model Output Directory")
                dl_btn     = gr.Button("📦  Prepare Zip", variant="secondary")

            dl_status = gr.Textbox(label="", interactive=False)
            dl_file   = gr.File(label="⬇️  Download Zip")

            gr.Markdown("---")
            gr.Markdown("### 🤗  Push to HuggingFace Hub")

            with gr.Row():
                hub_repo_inp  = gr.Textbox(label="Repo ID", placeholder="username/taoist-qwen-7b")
                hub_token_inp = gr.Textbox(label="HuggingFace Token", type="password", placeholder="hf_…")

            push_btn    = gr.Button("🚀  Push to Hub", variant="secondary")
            push_status = gr.Textbox(label="", interactive=False)

    # ── Events ───────────────────────────────────────────────
    train_btn.click(
        fn=start_and_stream,
        inputs=[
            base_model_inp, output_dir_inp, data_dir_inp,
            epochs_inp, batch_size_inp, grad_accum_inp, lora_r_inp,
            skip_scraping_inp, hf_token_inp,
        ],
        outputs=[log_box, status_md, train_btn],
    )

    stop_btn.click(fn=stop_training, outputs=status_md)

    dl_btn.click(
        fn=prepare_download,
        inputs=[dl_dir_inp],
        outputs=[dl_file, dl_status],
    )

    push_btn.click(
        fn=push_to_hub,
        inputs=[dl_dir_inp, hub_repo_inp, hub_token_inp],
        outputs=[push_status],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", 7860)),
        share=False,
        show_error=True,
    )
