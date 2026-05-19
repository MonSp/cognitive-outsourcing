import sys
import json

sys.path.insert(0, r"d:\trunk\SIG")
import co_benchmark as cb

plans = {}
for snum, plan in cb.PRECOMPUTED_PLANS.items():
    plans[str(snum)] = {
        "scenario": snum,
        "chain_of_thought": plan["chain_of_thought"],
        "nodes": plan["nodes"],
    }

with open(r"d:\trunk\SIG\co_benchmark_plans.json", "w", encoding="utf-8") as f:
    json.dump(plans, f, indent=2, ensure_ascii=False)

print("Done. Scenarios:", list(plans.keys()))
