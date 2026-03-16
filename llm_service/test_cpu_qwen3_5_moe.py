#!/usr/bin/env python3
"""Test Qwen3.5-35B-A3B-GPTQ-Int4 MoE model CPU inference."""

import sys
import os
import time

# Add the python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from llaisys.models import Qwen3_5Moe
from llaisys.libllaisys import DeviceType


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.5-35B-A3B-GPTQ-Int4"

    if not os.path.isdir(model_path):
        print(f"Model path not found: {model_path}")
        print("Usage: python test_cpu_qwen3_5_moe.py <model_path>")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    t0 = time.time()
    model = Qwen3_5Moe(model_path, DeviceType.CPU, 0)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Tokenize prompt
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception:
        print("Could not load tokenizer, using hardcoded token IDs")
        tokenizer = None

    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n"

    if tokenizer:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        print(f"Prompt tokens ({len(input_ids)}): {input_ids[:20]}...")
    else:
        # Fallback hardcoded
        input_ids = [248006, 77091, 198, 2610, 525, 264, 10950, 17847, 13, 248007, 198,
                     248006, 872, 198, 24948, 9468, 248007, 198, 248006, 77091, 198]

    print(f"\nPrompt: {prompt.strip()}")
    print(f"Generating (temperature=1.0, top_k=20)...")

    t0 = time.time()
    tokens = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=20,
    )
    elapsed = time.time() - t0

    if tokenizer:
        output = tokenizer.decode(tokens, skip_special_tokens=False)
    else:
        output = str(tokens)

    print(f"\nGenerated {len(tokens)} tokens in {elapsed:.1f}s ({len(tokens)/elapsed:.1f} tok/s)")
    print(f"Output tokens: {tokens}")
    print(f"Output text: {output}")

    # Greedy decode test
    print(f"\n--- Greedy decode ---")
    model.reset()
    t0 = time.time()
    tokens_greedy = model.generate(
        input_ids,
        max_new_tokens=20,
        temperature=0.0,
        top_k=1,
    )
    elapsed = time.time() - t0

    if tokenizer:
        output_greedy = tokenizer.decode(tokens_greedy, skip_special_tokens=False)
    else:
        output_greedy = str(tokens_greedy)

    print(f"Generated {len(tokens_greedy)} tokens in {elapsed:.1f}s")
    print(f"Output: {output_greedy}")


if __name__ == "__main__":
    main()
