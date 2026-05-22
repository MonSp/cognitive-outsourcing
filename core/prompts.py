"""Prompt templates for the Cognitive Outsourcing benchmark.

Collected from co_benchmark.py and r2_benchmark.py so that all prompt
strings live in one place and can be reused across experiment scripts.
"""

import re

SYSTEM_PROMPT = """You are a helpful travel assistant."""

SYSTEM_PROMPT_DEV = """You are an expert software developer."""

SIG_ANSWER_REMINDER = "\nBased on all the observations above, provide a comprehensive and accurate answer:"

TOOL_DESCRIPTIONS_TRAVEL = """Available tools:
1. search_attractions(city: str) - returns top attractions in the city
2. get_weather(city: str) - returns current weather
3. get_flight_info(origin: str, destination: str) - returns flight options"""

TOOL_DESCRIPTIONS_DEV = """Available tools:
1. run_test(test_name: str) - runs a test suite and returns the output
2. read_file(path: str) - returns the content of a source file
3. search_code(query: str) - searches codebase for relevant code snippets"""

TEACHER_PLANNING_PROMPT = """You are a cognitive planning expert. Given a user query, produce a chain-of-thought that includes marked nodes where tool results should be inserted.

{tool_descriptions}

Write a reasoning chain that demonstrates HOW to think about the problem and evaluate tool results. This chain will be given to a smaller local model, so you must teach it the reasoning process, not just list tool calls.

CRITICAL — Your chain-of-thought must include THREE elements for each tool call:
1. INTENT: Why you need this tool and what you expect to find
2. <<NODE:N>>: The tool call marker (result will be inserted here)
3. EVALUATION: After each node, evaluate the result — is it sufficient? What does it tell you? What should you do next based on this result?

IMPORTANT RULES:
- Write the chain-of-thought as the assistant's internal reasoning process.
- Insert <<NODE:N>> (1-indexed) at every point where a tool call is needed.
- Call each tool exactly once with correct arguments.
- Be precise with argument values (city names lowercase, no spaces: "newyork", "losangeles").
- After each <<NODE:N>>, include an EVALUATION of the result: assess completeness, note key facts, and reason about implications.
- If a result seems incomplete or problematic, note that and explain what it means for the answer.
- End with a synthesis that connects all evaluated results into a coherent answer plan.

OUTPUT FORMAT — respond with a single JSON object, nothing else:
{{
  "chain_of_thought": "I need to find attractions in Paris to help the user plan their trip. <<NODE:1>> The attractions list covers major landmarks like the Eiffel Tower and Louvre, which gives a good overview. Now I also need the weather to advise on packing, <<NODE:2>> The weather shows partly cloudy at 18C, which is mild and pleasant for sightseeing. With both the comprehensive attractions list and favorable weather, I can provide a well-rounded recommendation.",
  "nodes": {{
    "1": {{"tool": "search_attractions", "arguments": {{"city": "paris"}}}},
    "2": {{"tool": "get_weather", "arguments": {{"city": "paris"}}}}
  }}
}}"""

TEACHER_CONVERSATION_PROMPT = """You are a cognitive planning expert. Given a multi-turn conversation, produce a chain-of-thought for the ENTIRE conversation that includes marked nodes where tool results should be inserted.

{tool_descriptions}

Write a reasoning chain that demonstrates HOW to think about the problem and evaluate tool results. This chain will be given to a smaller local model, so you must teach it the reasoning process, not just list tool calls.

CRITICAL — Your chain-of-thought must include THREE elements for each tool call:
1. INTENT: Why you need this tool and what you expect to find
2. <<NODE:N>>: The tool call marker (result will be inserted here)
3. EVALUATION: After each node, evaluate the result — is it sufficient? What does it tell you? What should you do next based on this result?

IMPORTANT RULES:
- Write the chain-of-thought as the assistant's internal reasoning process covering all turns.
- Insert <<NODE:N>> (1-indexed) at every point where a tool call is needed.
- Call each tool exactly once with correct arguments.
- Be precise with argument values (city names lowercase, no spaces: "newyork", "losangeles").
- After each <<NODE:N>>, include an EVALUATION of the result: assess completeness, note key facts, and reason about implications.
- For conversational turns without tools, include natural reasoning without node markers.
- If a result seems incomplete or problematic, note that and explain what it means for the answer.
- End with a synthesis that connects all evaluated results into a coherent answer plan.

OUTPUT FORMAT — respond with a single JSON object, nothing else:
{{
  "chain_of_thought": "Turn 1: The user greets, I should respond warmly and offer help.\\nTurn 2: The user asks about Paris attractions, so I need to search for them, <<NODE:1>> The list includes the Eiffel Tower, Louvre, and other major sites — this is comprehensive enough to give good recommendations.\\nTurn 3: The user asks about weather, <<NODE:2>> The weather is partly cloudy at 18C, which is comfortable for outdoor sightseeing. I should mention this when recommending attractions.\\nWith the attractions list and pleasant weather, I can now provide a complete response.",
  "nodes": {{
    "1": {{"tool": "search_attractions", "arguments": {{"city": "paris"}}}},
    "2": {{"tool": "get_weather", "arguments": {{"city": "paris"}}}}
  }}
}}"""

RECALL_SYSTEM_PROMPT = """You are a helpful assistant. Answer the following question based ONLY on the information provided in the conversation above. If the information was mentioned, repeat it accurately. If you are not sure, say "I don't recall"."""

LOCAL_CO_PROMPT = """You are a helpful assistant. Based on all the observations below, provide a comprehensive and accurate answer.

Expert's reasoning: {reasoning}

Observations gathered:
{observations}

Now provide your answer:"""

NODE_PATTERN = re.compile(r'<<NODE:(\d+)>>')
