"""Test Qwen3.5-9B CPU inference."""

import sys
import time
sys.path.insert(0, "python")

import llaisys
from llaisys.libllaisys import DeviceType

MODEL_PATH = "models/qwen3_5_9b"

print("Creating Qwen3.5 model on CPU...")
t0 = time.time()
model = llaisys.models.Qwen3_5(MODEL_PATH, DeviceType.CPU, 0)
t1 = time.time()
print(f"Model created and weights loaded in {t1 - t0:.1f}s")

# Use tokenizers directly (doesn't need PyTorch >= 2.4)
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(f"{MODEL_PATH}/tokenizer.json")
print(f"Tokenizer loaded.")

# Use proper chat template
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\nHello"
encoded = tokenizer.encode(prompt)
input_ids = encoded.ids
print(f"Input tokens ({len(input_ids)}): {input_ids[:30]}...")
print(f"Input text: {repr(prompt)}")

# Run prefill + generate
print("\nGenerating (greedy, max 32 tokens)...")
t0 = time.time()
output_ids = []
for token_id in model.stream_generate(input_ids, max_new_tokens=32, top_k=1, temperature=0.0):
    output_ids.append(token_id)
    decoded = tokenizer.decode(output_ids)
    elapsed = time.time() - t0
    tok_per_s = len(output_ids) / elapsed if elapsed > 0 else 0
    sys.stdout.write(f"\r  [{len(output_ids):3d}] {tok_per_s:.2f} tok/s | {decoded[:80]}")
    sys.stdout.flush()
    if token_id in (model.eos_token_id, 248046):  # <|endoftext|> or <|im_end|>
        break

elapsed = time.time() - t0
print(f"\n\nGenerated {len(output_ids)} tokens in {elapsed:.1f}s ({len(output_ids)/elapsed:.2f} tok/s)")
print(f"Output: {tokenizer.decode(output_ids)}")
