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
    parser.add_argument("--model", default="models/qwen3_32b_fp8", type=str,
                        help="Path to model directory")
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--device-id", default=0, type=int,
                        help="GPU device ID (single GPU mode)")
    parser.add_argument("--tp", default=2, type=int,
                        help="Tensor parallelism degree (e.g. 2 for 2-GPU)")
    parser.add_argument("--max-tokens", default=20480, type=int,
                        help="Max new tokens per response")
    parser.add_argument("--system", default=None, type=str,
                        help="System prompt")
    parser.add_argument("--no-think", action="store_true",
                        help="Add /no_think to suppress thinking")
    parser.add_argument("--temperature", default=0.6, type=float,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", default=20, type=int,
                        help="Top-K sampling (1 = greedy, 0 = disabled)")
    parser.add_argument("--top-p", default=0.95, type=float,
                        help="Top-P (nucleus) sampling threshold")
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

    temp = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    print("=" * 60)
    print("  Qwen3 Chat  (llm_service)")
    print("  Commands:")
    print("    /clear        - clear conversation history")
    print("    /system       - set system prompt")
    print("    /tokens N     - set max tokens")
    print("    /temp F       - set temperature (0=greedy)")
    print("    /topk N       - set top-K (1=greedy, 0=off)")
    print("    /topp F       - set top-P")
    print("    /quit         - exit")
    print(f"  Sampling: temp={temp}, top_k={top_k}, top_p={top_p}")
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
            elif cmd[0] == "/temp":
                if len(cmd) > 1:
                    temp = float(cmd[1])
                print(f"[Temperature: {temp}]\n")
                continue
            elif cmd[0] == "/topk":
                if len(cmd) > 1:
                    top_k = int(cmd[1])
                print(f"[Top-K: {top_k}]\n")
                continue
            elif cmd[0] == "/topp":
                if len(cmd) > 1:
                    top_p = float(cmd[1])
                print(f"[Top-P: {top_p}]\n")
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
        new_ids = []
        in_think = False
        think_done = False
        think_buf = ""
        answer_buf = ""
        printed_header = False

        STOP_TAGS = {"<|im_end|>", "<|endoftext|>"}

        for token_id in model.stream_generate(
            input_ids, max_new_tokens=max_tokens,
            top_k=top_k, top_p=top_p, temperature=temp,
        ):
            new_ids.append(token_id)
            text = tokenizer.decode(new_ids, skip_special_tokens=False)

            if not think_done:
                if "<think>" in text and "</think>" in text:
                    parts = text.split("</think>", 1)
                    think_buf = parts[0].replace("<think>", "").strip()
                    think_done = True
                    if think_buf:
                        print(f"\033[90m[Think] {think_buf}\033[0m")
                    answer_part = parts[1] if len(parts) > 1 else ""
                    for tag in STOP_TAGS:
                        answer_part = answer_part.replace(tag, "")
                    if answer_part and not printed_header:
                        sys.stdout.write(f"\033[93mQwen3:\033[0m ")
                        printed_header = True
                    if len(answer_part) > len(answer_buf):
                        sys.stdout.write(answer_part[len(answer_buf):])
                        sys.stdout.flush()
                        answer_buf = answer_part
                    continue
                if "<think>" in text and not in_think:
                    in_think = True
                continue

            full_answer = text.split("</think>", 1)[-1] if "</think>" in text else text
            for tag in STOP_TAGS:
                full_answer = full_answer.replace(tag, "")

            if not printed_header and full_answer.strip():
                sys.stdout.write(f"\033[93mQwen3:\033[0m ")
                printed_header = True

            if len(full_answer) > len(answer_buf):
                sys.stdout.write(full_answer[len(answer_buf):])
                sys.stdout.flush()
                answer_buf = full_answer

        elapsed = time.time() - t0
        gen_tokens = len(new_ids)
        tps = gen_tokens / elapsed if elapsed > 0 else 0

        if not printed_header:
            full_text = tokenizer.decode(new_ids, skip_special_tokens=False)
            if "<think>" in full_text:
                parts = full_text.split("</think>", 1)
                think_buf = parts[0].replace("<think>", "").strip()
                if think_buf:
                    print(f"\033[90m[Think] {think_buf}\033[0m")
                answer_buf = parts[1].strip() if len(parts) > 1 else ""
            else:
                answer_buf = full_text
            for tag in STOP_TAGS:
                answer_buf = answer_buf.replace(tag, "")
            answer_buf = answer_buf.strip()
            print(f"\033[93mQwen3:\033[0m {answer_buf}")

        print(f"\n\033[90m({gen_tokens} tokens, {elapsed:.2f}s, {tps:.1f} tok/s)\033[0m\n")

        history.append(("assistant", answer_buf.strip()))


if __name__ == "__main__":
    main()
