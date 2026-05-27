"""Scenario builders for the Cognitive Outsourcing benchmark.

Extracted from co_benchmark.py to reduce file size and improve maintainability.
Each builder returns a list of turn dicts (user/tool/tool_args) or a tuple of
(system_prompt, turns).

Scenarios:
  1 — Long sequence (22 turns, cycling cities)
  2 — Multi-tool chain (5 turns, Paris→Rome)
  3 — Rapid-fire queries (12 turns, global cities)
  4 — Long document + tools (5 turns, travel guide background)
  5 — Mixed conversation (8 turns, chat + tools)
  6 — Deep tool chain (15 turns, round-the-world)
  7 — Travel planning chain (12 turns, NY→Tokyo)
  8 — Code debugging chain (5 turns, calculator bug)
  9 — Cross-reference chain (10 turns, 3-city comparison)
"""

from .prompts import SYSTEM_PROMPT


LONG_TRAVEL_GUIDE = """Comprehensive World Travel Guide — Background Reference
=====================================================
This document provides detailed travel information for major international destinations.
--- PARIS, FRANCE --- Paris, the capital of France, is one of the most visited cities in the world, attracting over 30 million tourists annually.
--- ROME, ITALY --- Rome, the Eternal City, has a history spanning over 2,800 years and serves as the capital of Italy.
--- TOKYO, JAPAN --- Tokyo, the capital of Japan, is the world's most populous metropolitan area with over 37 million residents.
--- LONDON, UNITED KINGDOM --- London, the capital of the United Kingdom, has a population of approximately 9 million.
--- NEW YORK CITY, USA --- New York City, the most populous city in the United States with over 8.3 million residents.
--- SYDNEY, AUSTRALIA --- Sydney is the largest city in Australia and Oceania, with a metropolitan population of over 5.3 million.
--- DUBAI, UNITED ARAB EMIRATES --- Dubai is the largest city in the UAE with a population of approximately 3.5 million.
--- BEIJING, CHINA --- Beijing, the capital of China, has a population of over 21 million and a history spanning over 3,000 years.
"""


def build_scenario1_long_sequence(n_turns=22):
    cities = ["paris", "rome", "tokyo", "london", "newyork", "sydney"]
    turns = []
    for i in range(n_turns):
        c1 = cities[i % len(cities)]
        c2 = cities[(i + 1) % len(cities)]
        if i % 3 == 0:
            turns.append({"user": f"What are the top attractions in {c1.title()}?",
                          "tool": "search_attractions", "tool_args": {"city": c1}})
        elif i % 3 == 1:
            turns.append({"user": f"How is the weather in {c1.title()}?",
                          "tool": "get_weather", "tool_args": {"city": c1}})
        else:
            turns.append({"user": f"Are there flights from {c1.title()} to {c2.title()}?",
                          "tool": "get_flight_info", "tool_args": {"origin": c1, "destination": c2}})
    return turns


def build_scenario2_multi_tool_chain():
    return [
        {"user": "I want to plan a trip from Paris to Rome. I need to know the attractions in both cities, the weather in Rome, and flight options. Help me with all of this.",
         "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "Continue finding the remaining information.", "tool": "search_attractions", "tool_args": {"city": "rome"}},
        {"user": "What about the weather?", "tool": "get_weather", "tool_args": {"city": "rome"}},
        {"user": "And the flights?", "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
        {"user": "Now give me a complete travel summary based on all the information you gathered.", "tool": None, "tool_args": None},
    ]


def build_scenario3_rapid_fire(n_queries=12):
    queries = [
        ("Weather in Paris?", "get_weather", {"city": "paris"}),
        ("Attractions in Tokyo?", "search_attractions", {"city": "tokyo"}),
        ("Flights London to Paris?", "get_flight_info", {"origin": "london", "destination": "paris"}),
        ("Weather in Dubai?", "get_weather", {"city": "dubai"}),
        ("Attractions in Rome?", "search_attractions", {"city": "rome"}),
        ("Flights NY to London?", "get_flight_info", {"origin": "newyork", "destination": "london"}),
        ("Weather in Sydney?", "get_weather", {"city": "sydney"}),
        ("Attractions in Beijing?", "search_attractions", {"city": "beijing"}),
        ("Flights Sydney to Tokyo?", "get_flight_info", {"origin": "sydney", "destination": "tokyo"}),
        ("Weather in London?", "get_weather", {"city": "london"}),
        ("Attractions in Dubai?", "search_attractions", {"city": "dubai"}),
        ("Flights Beijing to Dubai?", "get_flight_info", {"origin": "beijing", "destination": "dubai"}),
    ]
    return [{"user": q, "tool": t, "tool_args": a} for q, t, a in queries[:n_queries]]


def build_scenario4_long_document():
    system_with_doc = SYSTEM_PROMPT + "\n\n" + LONG_TRAVEL_GUIDE
    turns = [
        {"user": "Based on the background info, what are the attractions in Paris?", "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "What's the weather like there?", "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Any flights from New York to Tokyo?", "tool": "get_flight_info", "tool_args": {"origin": "newyork", "destination": "tokyo"}},
        {"user": "How about London to Dubai?", "tool": "get_flight_info", "tool_args": {"origin": "london", "destination": "dubai"}},
        {"user": "Summarize all the travel information you've gathered for me.", "tool": None, "tool_args": None},
    ]
    return system_with_doc, turns


def build_scenario5_mixed_conversation():
    return [
        {"user": "Hello! I'm thinking about traveling somewhere nice.", "tool": None, "tool_args": None},
        {"user": "What are the attractions in Paris?", "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "That sounds lovely! Is it expensive to visit?", "tool": None, "tool_args": None},
        {"user": "What's the weather like in Paris right now?", "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Great weather! I also love Italian food. Any attractions in Rome?", "tool": "search_attractions", "tool_args": {"city": "rome"}},
        {"user": "Can you compare Paris and Rome for me as travel destinations?", "tool": None, "tool_args": None},
        {"user": "Are there flights from Paris to Rome?", "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
        {"user": "Thanks! I think I'll visit both cities. Any packing tips?", "tool": None, "tool_args": None},
    ]


def build_scenario6_deep_tool_chain():
    return [
        {"user": "I'm planning a round-the-world trip starting from New York. First, I want to see attractions in New York.", "tool": "search_attractions", "tool_args": {"city": "newyork"}},
        {"user": "Great! Now check the weather in New York.", "tool": "get_weather", "tool_args": {"city": "newyork"}},
        {"user": "Next stop: London. What attractions are there?", "tool": "search_attractions", "tool_args": {"city": "london"}},
        {"user": "How's the weather in London?", "tool": "get_weather", "tool_args": {"city": "london"}},
        {"user": "Find me flights from New York to London.", "tool": "get_flight_info", "tool_args": {"origin": "newyork", "destination": "london"}},
        {"user": "After London, I want to visit Paris. What attractions are there?", "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "Check the weather in Paris too.", "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Find flights from London to Paris.", "tool": "get_flight_info", "tool_args": {"origin": "london", "destination": "paris"}},
        {"user": "Then I want to go to Dubai. Show me attractions there.", "tool": "search_attractions", "tool_args": {"city": "dubai"}},
        {"user": "What's the weather like in Dubai?", "tool": "get_weather", "tool_args": {"city": "dubai"}},
        {"user": "Find flights from Paris to Dubai.", "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "dubai"}},
        {"user": "Finally, I want to visit Tokyo. What are the top attractions?", "tool": "search_attractions", "tool_args": {"city": "tokyo"}},
        {"user": "Check Tokyo's weather as well.", "tool": "get_weather", "tool_args": {"city": "tokyo"}},
        {"user": "Find flights from Dubai to Tokyo.", "tool": "get_flight_info", "tool_args": {"origin": "dubai", "destination": "tokyo"}},
        {"user": "Now give me a complete round-the-world itinerary summary with all the information you gathered.", "tool": None, "tool_args": None},
    ]


def build_scenario7_travel_planning_chain():
    return [
        {"user": "I'm planning a trip from New York to Tokyo with stops in London and Dubai. What are the top attractions in New York?", "tool": "search_attractions", "tool_args": {"city": "newyork"}},
        {"user": "What's the weather like in New York right now?", "tool": "get_weather", "tool_args": {"city": "newyork"}},
        {"user": "Find me flights from New York to London please.", "tool": "get_flight_info", "tool_args": {"origin": "newyork", "destination": "london"}},
        {"user": "Now what attractions should I see in London?", "tool": "search_attractions", "tool_args": {"city": "london"}},
        {"user": "How's the weather in London?", "tool": "get_weather", "tool_args": {"city": "london"}},
        {"user": "I need flights from London to Dubai next.", "tool": "get_flight_info", "tool_args": {"origin": "london", "destination": "dubai"}},
        {"user": "What are the must-see attractions in Dubai?", "tool": "search_attractions", "tool_args": {"city": "dubai"}},
        {"user": "Tell me the weather in Dubai.", "tool": "get_weather", "tool_args": {"city": "dubai"}},
        {"user": "Find flights from Dubai to Tokyo.", "tool": "get_flight_info", "tool_args": {"origin": "dubai", "destination": "tokyo"}},
        {"user": "What are the best attractions in Tokyo?", "tool": "search_attractions", "tool_args": {"city": "tokyo"}},
        {"user": "What's the weather like in Tokyo?", "tool": "get_weather", "tool_args": {"city": "tokyo"}},
        {"user": "Based on all the weather information across all cities, recommend what I should pack for this trip.", "tool": None, "tool_args": None},
    ]


def build_scenario8_code_debugging_chain():
    return [
        {"user": "I have a bug in my Python project. The test_calculator test suite is failing. Can you run the test to see what's wrong?", "tool": "run_test", "tool_args": {"test_name": "test_calculator"}},
        {"user": "The test shows a failure. Can you read the calculator.py source code to understand the implementation?", "tool": "read_file", "tool_args": {"path": "calculator.py"}},
        {"user": "I see the issue might be in the divide method. Can you search the codebase for 'divide' to find all related code?", "tool": "search_code", "tool_args": {"query": "divide"}},
        {"user": "Can you also read the test_calculator.py file to see what's expected?", "tool": "read_file", "tool_args": {"path": "test_calculator.py"}},
        {"user": "Now that you have all the information, please explain the bug and suggest a fix.", "tool": None, "tool_args": None},
    ]


def build_scenario9_cross_reference_chain():
    return [
        {"user": "I want to compare travel options between Paris, Rome, and London. What are the top attractions in Paris?", "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "What's the weather like in Paris?", "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Now tell me about attractions in Rome.", "tool": "search_attractions", "tool_args": {"city": "rome"}},
        {"user": "How's the weather in Rome?", "tool": "get_weather", "tool_args": {"city": "rome"}},
        {"user": "What about attractions in London?", "tool": "search_attractions", "tool_args": {"city": "london"}},
        {"user": "And the weather in London?", "tool": "get_weather", "tool_args": {"city": "london"}},
        {"user": "Find me flights from Paris to Rome.", "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
        {"user": "How about flights from Rome to London?", "tool": "get_flight_info", "tool_args": {"origin": "rome", "destination": "london"}},
        {"user": "And flights from Paris to London?", "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "london"}},
        {"user": "Now please give me a comprehensive comparison of the three cities as travel destinations.", "tool": None, "tool_args": None},
    ]


BUILDERS = {
    1: build_scenario1_long_sequence,
    2: build_scenario2_multi_tool_chain,
    3: build_scenario3_rapid_fire,
    4: build_scenario4_long_document,
    5: build_scenario5_mixed_conversation,
    6: build_scenario6_deep_tool_chain,
    7: build_scenario7_travel_planning_chain,
    8: build_scenario8_code_debugging_chain,
    9: build_scenario9_cross_reference_chain,
}
