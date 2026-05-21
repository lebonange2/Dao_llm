#!/usr/bin/env python3
"""
Run the fine-tuned Taoist LLM locally.

Usage:
    python run_model.py --adapter ./adapter
    python run_model.py --adapter ./adapter --prompt "What is wu wei?"
    python run_model.py --adapter ./adapter --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(adapter_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    print(f"Loading base model: {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Applying LoRA adapter from: {adapter_path} ...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("✅ Model ready.\n")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 300):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def interactive_mode(model, tokenizer):
    print("=" * 60)
    print("  🌿 Taoist LLM — Interactive Mode")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n🌿 May your path align with the Tao.")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            print("\n🌿 May your path align with the Tao.")
            break

        response = generate(model, tokenizer, prompt)
        print(f"\n🌿 Dao: {response}")


def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuned Taoist LLM")
    parser.add_argument("--adapter", type=str, default="./adapter",
                        help="Path to the LoRA adapter folder")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model HuggingFace ID")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to answer (otherwise enters interactive mode)")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive chat mode")
    parser.add_argument("--max_tokens", type=int, default=300,
                        help="Max tokens to generate")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter, args.base_model)

    if args.prompt:
        response = generate(model, tokenizer, args.prompt, args.max_tokens)
        print(f"🌿 {response}")
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
