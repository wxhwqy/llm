import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "python")
import llaisys
from llaisys.libllaisys import DeviceType
from transformers import AutoTokenizer

MODEL = "models/qwen3_32b_fp8"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
print("Loading model TP=2...")
t0 = time.time()
model = llaisys.models.Qwen3(MODEL, DeviceType.NVIDIA, [0, 1])
print(f"Loaded in {time.time()-t0:.1f}s tp={model.tp_size}")
model.set_profile(True)
msgs = [{"role": "user", "content": "Hello, introduce yourself briefly /no_think"}]
txt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tokenizer.encode(txt)
print(f"Input: {len(ids)} tokens")
t0 = time.time()
out = model.generate(ids, max_new_tokens=256, top_k=1, top_p=1.0, temperature=1.0)
elapsed = time.time() - t0
new = out[len(ids):]
reply = tokenizer.decode(new, skip_special_tokens=True)
n = len(new)
print(f"Reply: {reply[:300]}")
print(f"Perf: {n} tok, {elapsed:.2f}s, {n/elapsed:.1f} tok/s")
