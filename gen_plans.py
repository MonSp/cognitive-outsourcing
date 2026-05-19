#!/usr/bin/env python3
"""
Generate co_benchmark_plans.json by calling a cloud LLM.

Usage:
  python gen_plans.py --api-base http://localhost:11434/v1 --model gpt-4o-mini
  python gen_plans.py --api-base https://api.openai.com/v1 --model gpt-4o-mini --api-key sk-xxx

Reads:  co_benchmark_prompts.json
Writes: co_benchmark_plans.json
"""

import argparse
import json
import os
import re
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. pip install requests")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_PATH = os.path.join(SCRIPT_DIR, "co_benchmark_prompts.json")
PLANS_PATH = os.path.join(SCRIPT_DIR, "co_benchmark_plans.json")


def parse_cot_plan(content: str) -> dict:
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        return {"chain_of_thought": "", "nodes": {}}
    try:
        plan = json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"chain_of_thought": "", "nodes": {}}

    cot = plan.get("chain_of_thought", plan.get("reasoning", ""))
    nodes = {}
    for key, val in plan.get("nodes", {}).items():
        name = val.get("tool") or val.get("name", "")
        args = val.get("arguments", {})
        if name and isinstance(args, dict):
            nodes[str(key)] = {"tool": name, "arguments": args}
    return {"chain_of_thought": cot, "nodes": nodes}


def call_llm(api_base: str, model: str, api_key: str,
             messages: list, max_tokens: int, temperature: float,
             timeout: float = 120.0) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate co_benchmark_plans.json from cloud LLM")
    parser.add_argument("--api-base", required=True,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:11434/v1)")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. gpt-4o-mini)")
    parser.add_argument("--api-key", default="",
                        help="API key (if required)")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-request timeout in seconds (default: 120)")
    parser.add_argument("--retry", type=int, default=2,
                        help="Retries on failure per scenario (default: 2)")
    parser.add_argument("--only", type=str, default="",
                        help="Only run specific scenarios, comma-separated (e.g. '1,3,7')")
    parser.add_argument("--skip", type=str, default="",
                        help="Skip specific scenarios, comma-separated (e.g. '3,5')")
    args = parser.parse_args()

    if not os.path.exists(PROMPTS_PATH):
        print(f"ERROR: {PROMPTS_PATH} not found")
        sys.exit(1)

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        all_prompts = json.load(f)

    only_set = set(int(x.strip()) for x in args.only.split(",") if x.strip())
    skip_set = set(int(x.strip()) for x in args.skip.split(",") if x.strip())

    existing_plans = {}
    if os.path.exists(PLANS_PATH):
        try:
            with open(PLANS_PATH, "r", encoding="utf-8") as f:
                existing_plans = json.load(f)
            print(f"[Info] Existing plans file loaded ({len(existing_plans)} scenarios)")
        except (json.JSONDecodeError, ValueError):
            print(f"[Warn] Existing plans file is empty or invalid, starting fresh")

    results = dict(existing_plans)
    failed = []

    for key in sorted(all_prompts.keys(), key=lambda k: int(k)):
        snum = int(key)
        if only_set and snum not in only_set:
            continue
        if snum in skip_set:
            continue

        p = all_prompts[key]
        name = p.get("name", f"Scenario {snum}")
        stype = p.get("type", "unknown")
        api_cfg = p.get("api_call", {})

        print(f"\n{'='*60}")
        print(f"Scenario {snum}: {name} (type={stype})")
        print(f"{'='*60}")

        plan = None
        for attempt in range(1 + args.retry):
            try:
                print(f"  Calling LLM (attempt {attempt+1})...", end=" ", flush=True)
                t0 = time.time()
                content = call_llm(
                    api_base=args.api_base,
                    model=args.model,
                    api_key=args.api_key,
                    messages=p["messages"],
                    max_tokens=api_cfg.get("max_tokens", 4096),
                    temperature=api_cfg.get("temperature", 0.0),
                    timeout=args.timeout,
                )
                elapsed = time.time() - t0
                print(f"OK ({elapsed:.1f}s, {len(content)} chars)")

                plan = parse_cot_plan(content)
                cot = plan["chain_of_thought"]
                nodes = plan["nodes"]

                if not cot or not nodes:
                    print(f"  WARNING: Empty chain_of_thought or nodes, retrying...")
                    plan = None
                    continue

                node_count = len(nodes)
                node_markers = len(re.findall(r'<<NODE:\d+>>', cot))
                print(f"  CoT: {len(cot)} chars, nodes: {node_count}, markers: {node_markers}")

                if node_count != node_markers:
                    print(f"  WARNING: Node count ({node_count}) != marker count ({node_markers})")

                break

            except requests.exceptions.Timeout:
                print(f"TIMEOUT")
            except requests.exceptions.ConnectionError:
                print(f"CONNECTION ERROR")
            except Exception as e:
                print(f"ERROR: {e}")

            plan = None

        if plan and plan["chain_of_thought"] and plan["nodes"]:
            results[key] = {
                "scenario": snum,
                "chain_of_thought": plan["chain_of_thought"],
                "nodes": plan["nodes"],
            }
            print(f"  -> SAVED")
        else:
            failed.append(snum)
            print(f"  -> FAILED (all attempts exhausted)")

    with open(PLANS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results written to {PLANS_PATH}")
    print(f"  Successful: {len(results)} scenarios")
    if failed:
        print(f"  Failed: {failed}")
    else:
        print(f"  All scenarios completed successfully!")


if __name__ == "__main__":
    main()
