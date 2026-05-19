import sys
import json

sys.path.insert(0, r"d:\trunk\SIG")
import co_benchmark as cb

prompts = {}

scenarios_multi = {
    1: ("Long-sequence stress test", cb.build_scenario1_long_sequence(22), cb.TOOL_DESCRIPTIONS_TRAVEL),
    2: ("Multi-tool chain", cb.build_scenario2_multi_tool_chain(), cb.TOOL_DESCRIPTIONS_TRAVEL),
    3: ("Rapid-fire short queries", cb.build_scenario3_rapid_fire(12), cb.TOOL_DESCRIPTIONS_TRAVEL),
    4: ("Long-document + tool calls", cb.build_scenario4_long_document()[1], cb.TOOL_DESCRIPTIONS_TRAVEL),
    5: ("Mixed conversation", cb.build_scenario5_mixed_conversation(), cb.TOOL_DESCRIPTIONS_TRAVEL),
    6: ("Deep tool chain", cb.build_scenario6_deep_tool_chain(), cb.TOOL_DESCRIPTIONS_TRAVEL),
    7: ("Travel planning (multi-turn)", cb.build_scenario7_travel_planning_chain(), cb.TOOL_DESCRIPTIONS_TRAVEL),
    8: ("Code debugging (multi-turn)", cb.build_scenario8_code_debugging_chain(), cb.TOOL_DESCRIPTIONS_DEV),
    9: ("Cross-reference analysis (multi-turn)", cb.build_scenario9_cross_reference_chain(), cb.TOOL_DESCRIPTIONS_TRAVEL),
}

for snum, (name, turns, tool_desc) in scenarios_multi.items():
    system_prompt = cb.CloudTeacherModule.TEACHER_CONVERSATION_PROMPT.format(
        tool_descriptions=tool_desc
    )
    conversation_text = ""
    for i, turn in enumerate(turns):
        conversation_text += "Turn {}: User: {}\n".format(i + 1, turn["user"])
    prompts[str(snum)] = {
        "scenario": snum,
        "name": name,
        "type": "conversation",
        "api_call": {
            "model": "gpt-4o-mini (or any OpenAI-compatible model)",
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_text.rstrip()},
        ],
    }

with open(r"d:\trunk\SIG\co_benchmark_prompts.json", "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=2, ensure_ascii=False)

print("Done. Scenarios:", list(prompts.keys()))
