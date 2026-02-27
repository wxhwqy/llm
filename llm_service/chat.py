#!/usr/bin/env python3
"""Interactive chat with Qwen3 via llm_service."""

import argparse
import sys
import io
import os
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

import llaisys
from llaisys.libllaisys import DeviceType
from transformers import AutoTokenizer


def build_messages(history, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for role, content in history:
        messages.append({"role": role, "content": content})
    return messages


def main():
    parser = argparse.ArgumentParser(description="Qwen3 Chat (llm_service)")
    parser.add_argument("--model", default="models/qwen3_14b_fp8", type=str,
                        help="Path to model directory")
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--device-id", default=0, type=int,
                        help="GPU device ID (single GPU mode)")
    parser.add_argument("--tp", default=1, type=int,
                        help="Tensor parallelism degree (e.g. 2 for 2-GPU)")
    parser.add_argument("--max-tokens", default=20480, type=int,
                        help="Max new tokens per response")
    parser.add_argument("--system", default=None, type=str,
                        help="System prompt")
    parser.add_argument("--no-think", action="store_true",
                        help="Add /no_think to suppress thinking")
    args = parser.parse_args()

    device = DeviceType.NVIDIA if args.device == "nvidia" else DeviceType.CPU

    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_DIR, model_path)
        if not os.path.isdir(model_path):
            model_path = os.path.abspath(args.model)

    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    if args.tp > 1:
        device_ids = list(range(args.tp))
        print(f"Loading Qwen3 model with TP={args.tp} on GPUs {device_ids} ...")
    else:
        device_ids = args.device_id
        print(f"Loading Qwen3 model on {args.device}:{args.device_id} ...")
    t0 = time.time()
    model = llaisys.models.Qwen3(model_path, device, device_ids)
    print(f"Model loaded in {time.time() - t0:.1f}s (tp={model.tp_size})\n")

    print("=" * 60)
    print("  Qwen3 Chat  (llm_service)")
    print("  Commands:")
    print("    /clear   - clear conversation history")
    print("    /system  - set system prompt")
    print("    /tokens  - set max tokens (e.g. /tokens 256)")
    print("    /quit    - exit")
    print("=" * 60)
    print()

    history = []
    system_prompt = args.system
    max_tokens = args.max_tokens

    while True:
        try:
            user_input = input("\033[96mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split(maxsplit=1)
            if cmd[0] == "/quit":
                print("Bye!")
                break
            elif cmd[0] == "/clear":
                history.clear()
                print("[History cleared]\n")
                continue
            elif cmd[0] == "/system":
                system_prompt = cmd[1] if len(cmd) > 1 else None
                print(f"[System prompt: {system_prompt or '(cleared)'}]\n")
                continue
            elif cmd[0] == "/tokens":
                if len(cmd) > 1:
                    max_tokens = int(cmd[1])
                print(f"[Max tokens: {max_tokens}]\n")
                continue

        if args.no_think:
            user_input += " /no_think"

        history.append(("user", user_input))
        messages = build_messages(history, system_prompt)

        input_text = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = tokenizer.encode(input_text)

        t0 = time.time()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        )
        elapsed = time.time() - t0

        new_ids = output_ids[len(input_ids):]
        reply = tokenizer.decode(new_ids, skip_special_tokens=False)

        think_text = ""
        answer_text = reply
        if "<think>" in reply:
            parts = reply.split("</think>", 1)
            if len(parts) == 2:
                think_text = parts[0].replace("<think>", "").strip()
                answer_text = parts[1].strip()
                for tag in ["<|im_end|>", "<|endoftext|>"]:
                    answer_text = answer_text.replace(tag, "")
                answer_text = answer_text.strip()
            else:
                answer_text = parts[0].replace("<think>", "").strip()
        else:
            for tag in ["<|im_end|>", "<|endoftext|>"]:
                answer_text = answer_text.replace(tag, "")
            answer_text = answer_text.strip()

        gen_tokens = len(new_ids)
        tps = gen_tokens / elapsed if elapsed > 0 else 0

        if think_text:
            print(f"\033[90m[Think] {think_text}\033[0m")

        print(f"\033[93mQwen3:\033[0m {answer_text}")
        print(f"\033[90m({gen_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s)\033[0m\n")

        history.append(("assistant", answer_text))


if __name__ == "__main__":
    main()
