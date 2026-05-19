#!/usr/bin/env python3
"""
Cognitive Outsourcing Benchmark — CO+AppLoop vs CO+SIG
=====================================================
Tests ONLY the Cognitive Outsourcing mode: cloud teacher plans, local model executes.
Three modes compared:
  1. co_apploop_full  : AppLoop (full re-prefill each turn) + CO planning
  2. co_apploop_gen  : AppLoop generation time only (excl. prefill) + CO planning
  3. co_sig          : SIG (KV-cache continuity) + CO planning

All modes share the same Phase-1 (cloud teacher plans tool chain).
Difference is ONLY in Phase-2 (local execution strategy).

Requires: pip install -r requirements.txt
  - llama-cpp-python >=0.2.80,<0.3.0
  - pynvml >=8.0.0
  - requests
"""

import time, json, argparse, warnings, re, logging, os
from typing import List, Dict, Optional, Tuple
from llama_cpp import Llama

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

SEQ_ID = 0

CITY_ALIASES = {
    "ny": "newyork", "new york": "newyork", "new york city": "newyork",
    "la": "losangeles", "sf": "sanfrancisco",
    "lv": "lasvegas", "dc": "washington",
}


def normalize_city(s: str) -> str:
    return CITY_ALIASES.get(s.strip().lower().replace(" ", ""), s.strip().lower().replace(" ", ""))


# ======================================================================
# GPU Monitor
# ======================================================================
class GPUMonitor:
    def __init__(self):
        self.handle = None
        self.baseline_mb = 0.0
        self.total_mb = 0.0
        self.enabled = False
        if not PYNVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_mb = info.total / (1024 ** 2)
            self.baseline_mb = info.used / (1024 ** 2)
            name = pynvml.nvmlDeviceGetName(self.handle)
            if isinstance(name, bytes):
                name = name.decode()
            print(f"[GPU] {name}, Total {self.total_mb:.0f} MB, Baseline {self.baseline_mb:.0f} MB")
        except Exception as e:
            print(f"[GPU] Init failed: {e}")

    def snapshot(self):
        if not self.enabled:
            return {"used_mb": 0.0, "delta_mb": 0.0}
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        used = info.used / (1024 ** 2)
        return {"used_mb": used, "delta_mb": used - self.baseline_mb}

    def shutdown(self):
        if self.enabled:
            pynvml.nvmlShutdown()


# ======================================================================
# MeaningCompiler — reused from temp.py
# ======================================================================
class MeaningCompiler:
    TOOL_MARK = "<<<TOOL>>>"
    TOOL_END = "<<</TOOL>>>"

    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = 4, n_gpu_layers: int = 0):
        print(f"Loading model {model_path} (ctx={n_ctx}, gpu_layers={n_gpu_layers})")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)
        self.n_ctx = n_ctx

    def tokenize(self, text: str, add_bos: bool = False) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def detokenize(self, ids: List[int]) -> str:
        return self.llm.detokenize(ids).decode("utf-8", errors="replace")

    def reset_cache(self):
        self.llm._ctx.kv_cache_seq_rm(SEQ_ID, -1, -1)
        self.llm.n_tokens = 0

    def eval(self, tokens: List[int]):
        self.llm.eval(tokens)

    def rebuild_cache(self, token_ids: List[int]):
        self.reset_cache()
        self.eval(token_ids)

    def sample(self, temp: float = 0.0) -> int:
        token = self.llm.sample(temp=temp)
        self.eval([token])
        return token

    def _ids_endswith(self, seq: List[int], suffix: List[int]) -> bool:
        if len(suffix) > len(seq):
            return False
        return list(seq[-len(suffix):]) == list(suffix)

    def _detect_repetition(self, text: str, min_len: int = 6, threshold: int = 3) -> bool:
        if threshold < 1 or len(text) < min_len:
            return False
        for pat_len in range(min_len, min(40, len(text) // threshold)):
            tail = text[-pat_len:]
            if tail.strip() == '':
                continue
            if text.count(tail) >= threshold:
                return True
        lines = text.split('\n')
        if len(lines) >= 3:
            last_line = lines[-1].strip()
            if len(last_line) > 5:
                recent = [l.strip() for l in lines[-4:]]
                if recent.count(last_line) >= 2:
                    return True
        return False

    def generate_until_ids(self, stop_ids: List[int], max_new: int = 300, rep_threshold: int = 3) -> Tuple[str, List[int]]:
        gen_ids = []
        for _ in range(max_new):
            token = self.sample()
            gen_ids.append(token)
            if self._ids_endswith(gen_ids, stop_ids):
                text_ids = gen_ids[:-len(stop_ids)]
                return self.detokenize(text_ids), gen_ids
            cur = self.detokenize(gen_ids)
            if "Assistant:" in cur or "assistant:" in cur:
                return cur, gen_ids
            if self._detect_repetition(cur, threshold=rep_threshold):
                break
        return self.detokenize(gen_ids), gen_ids

    def generate_until_str(self, stop_str: str, max_new: int = 300, rep_threshold: int = 3) -> Tuple[str, List[int]]:
        gen_ids = []
        for _ in range(max_new):
            token = self.sample()
            gen_ids.append(token)
            cur = self.detokenize(gen_ids)
            if stop_str in cur:
                return cur.split(stop_str)[0], gen_ids
            if "Assistant:" in cur or "assistant:" in cur:
                return cur, gen_ids
            if self._detect_repetition(cur, threshold=rep_threshold):
                break
        return self.detokenize(gen_ids), gen_ids

    def generate_until_any(self, stop_strs: List[str], max_new: int = 300, rep_threshold: int = 3) -> Tuple[str, List[int], Optional[str]]:
        all_stops = list(stop_strs) + ["Assistant:", "assistant:"]
        gen_ids = []
        for _ in range(max_new):
            token = self.sample()
            gen_ids.append(token)
            cur = self.detokenize(gen_ids)
            for s in all_stops:
                if s in cur:
                    return cur.split(s)[0], gen_ids, s
            if self._detect_repetition(cur, threshold=rep_threshold):
                break
        return self.detokenize(gen_ids), gen_ids, None

    def sanitize_generation(self, n_before: int, gen_text: str,
                            gen_ids: List[int], cached_prefix_ids: List[int]) -> Tuple[str, List[int], bool]:
        full_decoded = self.detokenize(gen_ids)
        need_rollback = False
        if "Assistant:" in full_decoded or "assistant:" in full_decoded.lower():
            need_rollback = True
        if not need_rollback and self._detect_repetition(full_decoded):
            need_rollback = True
        if not need_rollback:
            return gen_text, gen_ids, False
        self.rebuild_cache(cached_prefix_ids)
        return "", [], True


# ======================================================================
# InjectionEngine — reused from temp.py
# ======================================================================
class InjectionEngine:
    def __init__(self, compiler: MeaningCompiler):
        self.compiler = compiler
        self.cached_ids: List[int] = []

    def make_result_block(self, tool_name: str, tool_args: dict, result: str) -> str:
        args_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
        return (
            f"{self.compiler.TOOL_END}\n"
            f"[Observation from {tool_name}({args_str})]: {result}\n"
        )

    def inject(self, token_ids: List[int]) -> Tuple[int, float]:
        t0 = time.time()
        self.compiler.eval(token_ids)
        elapsed = time.time() - t0
        return len(token_ids), elapsed

    def inject_and_track(self, token_ids: List[int], metrics: Dict, key_prefix: str = "total"):
        n_tok, elapsed = self.inject(token_ids)
        metrics[f"{key_prefix}_prefill_tokens"] += n_tok
        metrics[f"{key_prefix}_prefill_time"] += elapsed
        self.cached_ids = self.cached_ids + list(token_ids)

    def update_cache(self, new_ids: List[int]):
        self.cached_ids = self.cached_ids + list(new_ids)

    def rollback(self, target_ids: List[int]):
        self.compiler.rebuild_cache(target_ids)
        self.cached_ids = list(target_ids)

    def reset(self):
        self.compiler.reset_cache()
        self.cached_ids = []


# ======================================================================
# Tool Registry — copied from temp.py
# ======================================================================
class ToolRegistry:
    def __init__(self):
        self.tools = {
            "search_attractions": {
                "paris": "1. Eiffel Tower (330m). 2. Louvre (Mona Lisa). 3. Notre-Dame. 4. Montmartre. 5. Seine Cruise.",
                "rome": "1. Colosseum. 2. Vatican City. 3. Trevi Fountain. 4. Roman Forum. 5. Pantheon.",
                "tokyo": "1. Senso-ji. 2. Shibuya Crossing. 3. Meiji Shrine. 4. Akihabara. 5. Tokyo Skytree.",
                "london": "1. British Museum. 2. Tower of London. 3. London Eye. 4. Buckingham Palace. 5. Big Ben.",
                "newyork": "1. Statue of Liberty. 2. Central Park. 3. Empire State Building. 4. Times Square. 5. Broadway.",
                "sydney": "1. Opera House. 2. Harbour Bridge. 3. Bondi Beach. 4. Taronga Zoo. 5. The Rocks.",
                "beijing": "1. Great Wall. 2. Forbidden City. 3. Temple of Heaven. 4. Summer Palace. 5. Tiananmen Square.",
                "dubai": "1. Burj Khalifa. 2. Palm Jumeirah. 3. Dubai Mall. 4. Dubai Marina. 5. Jumeirah Beach.",
            },
            "get_weather": {
                "paris": "Partly cloudy, 18C",
                "rome": "Sunny, 26C",
                "tokyo": "Rain, 22C",
                "london": "Overcast, 15C",
                "newyork": "Clear, 22C",
                "sydney": "Sunny, 24C",
                "beijing": "Smog, 28C",
                "dubai": "Hot, 38C",
            },
            "get_flight_info": {
                "paris_rome": "Direct: Air France AF1104 (2h10m, $180)",
                "paris_tokyo": "Direct: Air France AF276 (11h40m, $850)",
                "rome_tokyo": "Connecting via Istanbul (15h, $720)",
                "london_paris": "Eurostar train (2h20m, $120) or flights",
                "newyork_london": "Direct: BA178 (7h, $600)",
                "sydney_tokyo": "Direct: JQ26 (9h, $450)",
                "beijing_dubai": "Direct: EK307 (8h, $700)",
                "newyork_tokyo": "Direct: JL5 (13h, $900)",
                "london_dubai": "Direct: EK1 (7h, $650)",
                "paris_newyork": "Direct: AF8 (8h30m, $550)",
                "dubai_tokyo": "Direct: EK318 (9h30m, $800)",
            },
            "read_file": {
                "calculator.py": "class Calculator:\n    def __init__(self):\n        self.history = []\n    def add(self, a, b):\n        result = a + b\n        self.history.append(f'{a}+{b}={result}')\n        return result\n    def subtract(self, a, b):\n        result = a - b\n        self.history.append(f'{a}-{b}={result}')\n        return result\n    def multiply(self, a, b):\n        result = a * b\n        self.history.append(f'{a}*{b}={result}')\n        return result\n    def divide(self, a, b):\n        result = a / b\n        self.history.append(f'{a}/{b}={result}')\n        return result\n    def power(self, a, b):\n        result = a ** b\n        self.history.append(f'{a}**{b}={result}')\n        return result",
                "test_calculator.py": "import unittest\nfrom calculator import Calculator\n\nclass TestCalculator(unittest.TestCase):\n    def setUp(self):\n        self.calc = Calculator()\n    def test_add(self):\n        self.assertEqual(self.calc.add(2, 3), 5)\n    def test_subtract(self):\n        self.assertEqual(self.calc.subtract(5, 3), 2)\n    def test_multiply(self):\n        self.assertEqual(self.calc.multiply(4, 3), 12)\n    def test_divide(self):\n        self.assertEqual(self.calc.divide(10, 2), 5)\n    def test_divide_by_zero(self):\n        with self.assertRaises(ZeroDivisionError):\n            self.calc.divide(10, 0)\n    def test_power(self):\n        self.assertEqual(self.calc.power(2, 3), 8)\n    def test_history(self):\n        self.calc.add(1, 2)\n        self.calc.multiply(3, 4)\n        self.assertEqual(len(self.calc.history), 2)",
                "auth.py": "import hashlib\nimport secrets\n\nclass AuthManager:\n    def __init__(self):\n        self.users = {}\n    def register(self, username, password):\n        salt = secrets.token_hex(16)\n        hashed = hashlib.sha256((password + salt).encode()).hexdigest()\n        self.users[username] = {'salt': salt, 'hash': hashed}\n        return True\n    def login(self, username, password):\n        if username not in self.users:\n            return False\n        user = self.users[username]\n        hashed = hashlib.sha256((password + user['salt']).encode()).hexdigest()\n        return hashed == user['hash']\n    def change_password(self, username, old_pw, new_pw):\n        if not self.login(username, old_pw):\n            return False\n        salt = secrets.token_hex(16)\n        hashed = hashlib.sha256((new_pw + salt).encode()).hexdigest()\n        self.users[username] = {'salt': salt, 'hash': hashed}\n        return True",
                "config.py": "import json\nimport os\n\nCONFIG_PATH = '/etc/app/config.json'\nDEFAULT_CONFIG = {\n    'debug': False,\n    'port': 8080,\n    'database': {'host': 'localhost', 'port': 5432, 'name': 'app_db'},\n    'cache': {'enabled': True, 'ttl': 3600},\n    'logging': {'level': 'INFO', 'file': '/var/log/app.log'}\n}\n\ndef load_config():\n    if os.path.exists(CONFIG_PATH):\n        with open(CONFIG_PATH) as f:\n            return json.load(f)\n    return DEFAULT_CONFIG.copy()\n\ndef get_db_url(config=None):\n    cfg = config or load_config()\n    db = cfg['database']\n    return f\"postgresql://{db['host']}:{db['port']}/{db['name']}\"",
                "api.py": "from flask import Flask, jsonify, request\nfrom auth import AuthManager\nfrom config import load_config\n\napp = Flask(__name__)\nauth = AuthManager()\nconfig = load_config()\n\n@app.route('/api/register', methods=['POST'])\ndef register():\n    data = request.json\n    return jsonify({'success': auth.register(data['username'], data['password'])})\n\n@app.route('/api/login', methods=['POST'])\ndef login():\n    data = request.json\n    return jsonify({'success': auth.login(data['username'], data['password'])})\n\n@app.route('/api/config', methods=['GET'])\ndef get_config():\n    return jsonify(config)\n\n@app.route('/api/health', methods=['GET'])\ndef health():\n    return jsonify({'status': 'ok', 'debug': config.get('debug', False)})",
            },
            "search_code": {
                "divide": "Found in calculator.py line 14: def divide(self, a, b): result = a / b  -- NOTE: No zero-division check!",
                "login": "Found in auth.py line 12: def login(self, username, password): -- Uses SHA256 with salt",
                "config": "Found in config.py: DEFAULT_CONFIG with database, cache, logging settings",
                "register": "Found in auth.py line 7: def register(self, username, password): -- Stores salt+hash",
                "api_route": "Found in api.py: Routes /api/register, /api/login, /api/config, /api/health",
                "database": "Found in config.py: database.host=localhost, database.port=5432, database.name=app_db",
                "error_handling": "No try/except blocks found in any file. Missing error handling in divide() and API routes.",
                "test": "Found in test_calculator.py: 7 test cases covering add, subtract, multiply, divide, divide_by_zero, power, history",
            },
            "run_test": {
                "test_calculator": "FAILED: test_divide_by_zero - Expected ZeroDivisionError but got 5.0 (divide(10,0) returned inf instead of raising)\nPASSED: test_add - 2+3=5\nPASSED: test_subtract - 5-3=2\nPASSED: test_multiply - 4*3=12\nPASSED: test_divide - 10/2=5\nPASSED: test_power - 2**3=8\nPASSED: test_history\n1 FAILED, 6 PASSED out of 7 tests",
                "test_auth": "PASSED: test_register\nPASSED: test_login_success\nPASSED: test_login_wrong_password\nPASSED: test_change_password\n4 PASSED out of 4 tests",
                "test_api": "FAILED: test_health_endpoint - Expected status=ok, got 500 Internal Server Error\nFAILED: test_config_endpoint - Missing 'cache' key in response\nPASSED: test_register_endpoint\nPASSED: test_login_endpoint\n2 FAILED, 2 PASSED out of 4 tests",
                "test_all": "TOTAL: 3 FAILED, 12 PASSED out of 15 tests\nFailures: test_divide_by_zero, test_health_endpoint, test_config_endpoint",
            },
        }

    def execute(self, name: str, args: dict) -> str:
        if name == "search_attractions":
            key = args.get("city", "").strip().lower()
            return self.tools["search_attractions"].get(key, f"No data for {key}")
        elif name == "get_weather":
            key = args.get("city", "").strip().lower()
            return self.tools["get_weather"].get(key, f"No data for {key}")
        elif name == "get_flight_info":
            o = args.get("origin", "").strip().lower()
            d = args.get("destination", "").strip().lower()
            return self.tools["get_flight_info"].get(f"{o}_{d}", f"No flights between {o} and {d}")
        elif name == "read_file":
            key = args.get("path", args.get("file", "")).strip()
            return self.tools["read_file"].get(key, f"File not found: {key}")
        elif name == "search_code":
            key = args.get("query", "").strip().lower()
            return self.tools["search_code"].get(key, f"No results for '{key}'")
        elif name == "run_test":
            key = args.get("test_name", args.get("name", "")).strip()
            return self.tools["run_test"].get(key, f"Test not found: {key}")
        return f"Unknown tool: {name}"


# ======================================================================
# Cloud Teacher — only the planning part, reused from temp.py
# ======================================================================
class CloudTeacherModule:
    def __init__(self, api_base: str = "http://localhost:11434/v1",
                 model: str = "gpt-4o-mini",
                 api_key: str = "",
                 timeout: float = 30.0):
        if not REQUESTS_AVAILABLE:
            raise ImportError("CloudTeacherModule requires 'requests': pip install requests")
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.logger = logging.getLogger("CloudTeacher")

    TOOL_DESCRIPTIONS_TRAVEL = """Available tools:
1. search_attractions(city: str) - returns top attractions in the city
2. get_weather(city: str) - returns current weather
3. get_flight_info(origin: str, destination: str) - returns flight options"""

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

    def plan_tool_chain(self, query: str, tool_descriptions: str = None) -> Dict:
        if tool_descriptions is None:
            tool_descriptions = self.TOOL_DESCRIPTIONS_TRAVEL
        prompt = self.TEACHER_PLANNING_PROMPT.format(tool_descriptions=tool_descriptions)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            "max_tokens": 2048,
            "temperature": 0.0,
        }

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            self.logger.info(f"Teacher plan raw: {content[:200]}...")
            return self._parse_cot_plan(content)
        except requests.exceptions.Timeout:
            self.logger.warning("CloudTeacher planning timeout")
        except requests.exceptions.ConnectionError:
            self.logger.warning(f"CloudTeacher connection error during planning")
        except Exception as e:
            self.logger.warning(f"CloudTeacher planning error: {e}")
        return {"chain_of_thought": "", "nodes": {}}

    def plan_conversation(self, turns: List[Dict],
                          tool_descriptions: str = None) -> Dict:
        if tool_descriptions is None:
            tool_descriptions = self.TOOL_DESCRIPTIONS_TRAVEL
        prompt = self.TEACHER_CONVERSATION_PROMPT.format(tool_descriptions=tool_descriptions)

        conversation_text = ""
        for i, turn in enumerate(turns):
            conversation_text += f"Turn {i+1}: User: {turn['user']}\n"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": conversation_text},
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            self.logger.info(f"Teacher conv plan raw: {content[:200]}...")
            return self._parse_cot_plan(content)
        except requests.exceptions.Timeout:
            self.logger.warning("CloudTeacher conversation planning timeout")
        except requests.exceptions.ConnectionError:
            self.logger.warning("CloudTeacher connection error during conversation planning")
        except Exception as e:
            self.logger.warning(f"CloudTeacher conversation planning error: {e}")
        return {"chain_of_thought": "", "nodes": {}}

    def _parse_cot_plan(self, content: str) -> Dict:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            self.logger.warning("No JSON found in teacher response")
            return {"chain_of_thought": "", "nodes": {}}
        try:
            plan = json.loads(json_match.group())
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse teacher JSON")
            return {"chain_of_thought": "", "nodes": {}}

        cot = plan.get("chain_of_thought", plan.get("reasoning", ""))
        nodes = {}
        for key, val in plan.get("nodes", {}).items():
            name = val.get("tool") or val.get("name", "")
            args = val.get("arguments", {})
            if name and isinstance(args, dict):
                nodes[str(key)] = {"tool": name, "arguments": args}
        return {"chain_of_thought": cot, "nodes": nodes}


# ======================================================================
# System Prompts
# ======================================================================
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

LOCAL_CO_PROMPT = """You are a helpful assistant. Based on all the observations below, provide a comprehensive and accurate answer.

Expert's reasoning: {reasoning}

Observations gathered:
{observations}

Now provide your answer:"""


# ======================================================================
# CO Agents: two modes share the same Phase-1 (cloud planning)
#            differ ONLY in Phase-2 (local execution strategy)
# ======================================================================

def _init_metrics() -> Dict:
    return {
        "total_ttf": 0.0,
        "total_gen_time": 0.0,
        "total_prefill_time": 0.0,
        "per_turn_ttf": [],
        "tool_turn_ttf": [],
        "chat_turn_ttf": [],
        "tool_calls_ok": 0,
        "total_tool_calls": 0,
        "final_answer": "",
        "peak_gpu_delta": 0.0,
        "total_gen_tokens": 0,
        "total_prefill_tokens": 0,
        "chain_depth": 0,
        "chain_total": 0,
        "tool_results_text": "",
    }


def _make_observation_block(compiler: MeaningCompiler,
                             tool_name: str, tool_args: dict,
                             result: str) -> str:
    args_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
    return f"\n[Observation from {tool_name}({args_str})]: {result}"


NODE_PATTERN = re.compile(r'<<NODE:(\d+)>>')


def assemble_chain_of_thought(cot: str, nodes: Dict,
                               module: ToolRegistry,
                               expected_chain: List[Dict],
                               metrics: Dict,
                               debug: bool = True) -> Tuple[str, int]:
    """
    Core of the Cognitive Outsourcing agent-level assembly:
    1. Find all <<NODE:N>> markers in the teacher's chain-of-thought
    2. For each node, call the corresponding tool
    3. Replace <<NODE:N>> with the tool result
    4. Return the assembled chain with results filled in

    Returns: (assembled_cot, matched_count)
    """
    matched_count = 0
    matched_flags = [False] * len(expected_chain)

    def replace_node(match):
        nonlocal matched_count
        node_id = match.group(1)
        if node_id not in nodes:
            if debug:
                print(f"     [NODE:{node_id}] Not found in plan, skipping")
            return f"[Node {node_id}: no tool specified]"

        node = nodes[node_id]
        tool_name = node["tool"]
        tool_args = node["arguments"]
        tool_result = module.execute(tool_name, tool_args)

        for j, expected in enumerate(expected_chain):
            if matched_flags[j]:
                continue
            true_name = expected["tool"]
            true_args = expected["tool_args"]
            name_ok = (tool_name == true_name)
            args_ok = False
            if tool_args and true_args:
                normalized_parsed = {k: normalize_city(str(v)) for k, v in tool_args.items()}
                normalized_true = {k: normalize_city(str(v)) for k, v in true_args.items()}
                args_ok = (normalized_parsed == normalized_true)
            if name_ok and args_ok:
                matched_count += 1
                matched_flags[j] = True
                break

        metrics["total_tool_calls"] += 1

        if debug:
            print(f"     [NODE:{node_id}] {tool_name}({json.dumps(tool_args)}) → {tool_result[:60]}...")

        return f"\n[Result of {tool_name}({', '.join(f'{k}={v}' for k, v in tool_args.items())})]: {tool_result}\n"

    assembled = NODE_PATTERN.sub(replace_node, cot)
    metrics["tool_calls_ok"] += matched_count
    return assembled, matched_count


# ======================================================================
# System Prompts
# ======================================================================


# --------------------------------------------------------------------------
# CO + AppLoop (full re-prefill each turn)
# --------------------------------------------------------------------------
class COAppLoopAgent:
    """
    Phase-1: Cloud teacher produces a chain-of-thought with <<NODE:N>> markers.
    Phase-2: Agent script assembles tool results into the CoT, then for each
             turn, full re-prefills the entire conversation history + assembled
             CoT, and generates the answer.
    """

    def __init__(self, compiler: MeaningCompiler, module: ToolRegistry,
                 teacher: Optional[CloudTeacherModule] = None, max_new: int = 600,
                 max_new_tool: int = 300):
        self.compiler = compiler
        self.module = module
        self.teacher = teacher
        self.max_new = max_new
        self.max_new_tool = max_new_tool

    def _full_prefill(self, text: str, metrics: Dict) -> List[int]:
        full_ids = list(self.compiler.tokenize(text, add_bos=False))
        self.compiler.reset_cache()
        pf_t0 = time.time()
        self.compiler.eval(full_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        metrics["total_prefill_tokens"] += len(full_ids)
        return list(full_ids)

    # ---- Multi-turn (Scenarios 1-6) ----

    def run_conversation(self, turns: List[Dict],
                         system_prompt: str = SYSTEM_PROMPT,
                         tool_descriptions: str = TOOL_DESCRIPTIONS_TRAVEL,
                         gpu: Optional[GPUMonitor] = None,
                         debug: bool = True,
                         precomputed_plan: Optional[Dict] = None) -> Dict:
        metrics = _init_metrics()

        if precomputed_plan:
            plan = precomputed_plan
        else:
            plan = self.teacher.plan_conversation(turns, tool_descriptions=tool_descriptions)

        cot = plan.get("chain_of_thought", "")
        nodes = plan.get("nodes", {})

        expected_chain = [t for t in turns if t.get("tool")]
        assembled_cot, _ = assemble_chain_of_thought(
            cot, nodes, self.module, expected_chain, metrics, debug=debug)

        if debug:
            print(f"   Assembled CoT ({len(assembled_cot)} chars, {len(nodes)} nodes):")
            print(f"   {assembled_cot[:300]}...")

        history = f"{system_prompt}\n\n"
        prev_ids = self._full_prefill(history, metrics)

        if gpu:
            metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        cot_injected = False
        cot_token_count = len(self.compiler.tokenize(assembled_cot, add_bos=False))

        for i, turn in enumerate(turns):
            t0 = time.time()
            history += f"User: {turn['user']}\nAssistant:"

            is_cot_turn = turn.get("tool") and not cot_injected
            if is_cot_turn:
                history += f"\n{assembled_cot}\n\nAnswer:"
                cot_injected = True

            prev_ids = self._full_prefill(history, metrics)

            if turn.get("tool") and not is_cot_turn:
                cur_max_new = self.max_new_tool
            elif is_cot_turn:
                cur_max_new = max(self.max_new * 2 // 3, self.max_new - cot_token_count // 6)
            else:
                cur_max_new = self.max_new

            gen_t0 = time.time()
            gen_text, gen_ids = self.compiler.generate_until_str("\nUser:", max_new=cur_max_new)
            metrics["total_gen_time"] += time.time() - gen_t0
            metrics["total_gen_tokens"] += len(gen_ids)
            metrics["final_answer"] = gen_text.strip()
            history += gen_text + "\n"
            prev_ids = prev_ids + list(gen_ids)

            ttf = time.time() - t0
            metrics["per_turn_ttf"].append(ttf)
            if turn.get("tool"):
                metrics["tool_turn_ttf"].append(ttf)
            else:
                metrics["chat_turn_ttf"].append(ttf)
            if gpu:
                metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        metrics["total_ttf"] = sum(metrics["per_turn_ttf"])
        metrics["tool_results_text"] = assembled_cot
        return metrics

    # ---- Complex task (Scenarios 7-9) ----

    def run_complex_task(self, user_query: str, expected_chain: List[Dict],
                         system_prompt: str = SYSTEM_PROMPT,
                         tool_descriptions: str = TOOL_DESCRIPTIONS_TRAVEL,
                         gpu: Optional[GPUMonitor] = None,
                         debug: bool = True,
                         precomputed_plan: Optional[Dict] = None) -> Dict:
        metrics = _init_metrics()
        metrics["chain_total"] = len(expected_chain)

        if precomputed_plan:
            plan = precomputed_plan
        else:
            plan = self.teacher.plan_tool_chain(user_query, tool_descriptions=tool_descriptions)

        cot = plan.get("chain_of_thought", "")
        nodes = plan.get("nodes", {})

        if debug:
            print(f"   Teacher CoT ({len(cot)} chars, {len(nodes)} nodes):")
            print(f"   {cot[:300]}...")

        assembled_cot, matched = assemble_chain_of_thought(
            cot, nodes, self.module, expected_chain, metrics, debug=debug)
        metrics["chain_depth"] = matched

        if debug:
            print(f"   Assembled CoT ({len(assembled_cot)} chars):")
            print(f"   {assembled_cot[:300]}...")

        full_text = f"{system_prompt}\n\nUser: {user_query}\nAssistant:\n{assembled_cot}\n\nAnswer:"

        self._full_prefill(full_text, metrics)

        cot_token_count = len(self.compiler.tokenize(assembled_cot, add_bos=False))
        cot_turn_max_new = max(self.max_new * 2 // 3, self.max_new - cot_token_count // 6)

        gen_t0 = time.time()
        gen_text, gen_ids = self.compiler.generate_until_str("\nUser:", max_new=cot_turn_max_new)
        metrics["total_gen_time"] += time.time() - gen_t0
        metrics["total_gen_tokens"] += len(gen_ids)
        metrics["total_ttf"] = metrics["total_gen_time"] + metrics["total_prefill_time"]
        metrics["per_turn_ttf"] = [metrics["total_ttf"]]
        metrics["final_answer"] = gen_text.strip()
        metrics["tool_results_text"] = assembled_cot

        if debug:
            print(f"   Final answer:\n{gen_text.strip()[:300]}")

        if gpu:
            metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        return metrics


# --------------------------------------------------------------------------
# CO + SIG (KV-cache continuity)
# --------------------------------------------------------------------------
class COSIGAgent:
    """
    Phase-1: Cloud teacher produces a chain-of-thought with <<NODE:N>> markers.
    Phase-2: Agent script assembles tool results into the CoT, then injects
             the assembled CoT into the KV cache (no re-prefill), and the
             local model continues generating from where the CoT left off.

    Key difference from COAppLoop: KV-cache is maintained. The assembled CoT
    is injected once, and the model simply continues the generation.
    """

    def __init__(self, compiler: MeaningCompiler, module: ToolRegistry,
                 teacher: Optional[CloudTeacherModule] = None, max_new: int = 600,
                 max_new_tool: int = 300, rep_threshold: int = 2,
                 max_new_tool_sig: int = 150):
        self.compiler = compiler
        self.module = module
        self.teacher = teacher
        self.max_new = max_new
        self.max_new_tool = max_new_tool
        self.max_new_tool_sig = max_new_tool_sig
        self.rep_threshold = rep_threshold
        self.engine = InjectionEngine(compiler)

    # ---- Multi-turn (Scenarios 1-6) ----

    def run_conversation(self, turns: List[Dict],
                         system_prompt: str = SYSTEM_PROMPT,
                         tool_descriptions: str = TOOL_DESCRIPTIONS_TRAVEL,
                         gpu: Optional[GPUMonitor] = None,
                         debug: bool = True,
                         precomputed_plan: Optional[Dict] = None) -> Dict:
        metrics = _init_metrics()
        self.engine.reset()

        if precomputed_plan:
            plan = precomputed_plan
        else:
            plan = self.teacher.plan_conversation(turns, tool_descriptions=tool_descriptions)

        cot = plan.get("chain_of_thought", "")
        nodes = plan.get("nodes", {})

        expected_chain = [t for t in turns if t.get("tool")]
        assembled_cot, _ = assemble_chain_of_thought(
            cot, nodes, self.module, expected_chain, metrics, debug=debug)

        if debug:
            print(f"   Assembled CoT ({len(assembled_cot)} chars, {len(nodes)} nodes):")
            print(f"   {assembled_cot[:300]}...")

        history = f"{system_prompt}\n\n"
        init_ids = list(self.compiler.tokenize(history, add_bos=False))

        pf_t0 = time.time()
        self.compiler.eval(init_ids)
        metrics["total_prefill_tokens"] += len(init_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        self.engine.update_cache(init_ids)

        if gpu:
            metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        cot_injected = False
        cot_token_count = len(self.compiler.tokenize(assembled_cot, add_bos=False))

        for i, turn in enumerate(turns):
            t0 = time.time()
            user_line = f"User: {turn['user']}\nAssistant:"
            user_ids = list(self.compiler.tokenize(user_line, add_bos=False))
            pf_t0 = time.time()
            self.compiler.eval(user_ids)
            metrics["total_prefill_tokens"] += len(user_ids)
            metrics["total_prefill_time"] += time.time() - pf_t0
            self.engine.update_cache(user_ids)

            is_cot_turn = turn.get("tool") and not cot_injected
            if turn.get("tool") and not is_cot_turn:
                cur_max_new = self.max_new_tool_sig
                cur_rep_threshold = self.rep_threshold
            elif is_cot_turn:
                cur_max_new = max(self.max_new * 2 // 3, self.max_new - cot_token_count // 6)
                cur_rep_threshold = 3
            else:
                cur_max_new = self.max_new
                cur_rep_threshold = 3

            if is_cot_turn:
                cot_block = f"\n{assembled_cot}\n\nAnswer:"
                cot_ids = list(self.compiler.tokenize(cot_block, add_bos=False))
                self.engine.inject_and_track(cot_ids, metrics)
                cot_injected = True

                gen_t0 = time.time()
                gen_text, gen_ids = self.compiler.generate_until_str("\nUser:", max_new=cur_max_new, rep_threshold=cur_rep_threshold)
                metrics["total_gen_time"] += time.time() - gen_t0
                metrics["total_gen_tokens"] += len(gen_ids)
                self.engine.update_cache(list(gen_ids))
                metrics["final_answer"] = gen_text.strip()
            else:
                is_last_turn = (i == len(turns) - 1)
                if not turn.get("tool") and is_last_turn:
                    reminder_ids = list(self.compiler.tokenize(SIG_ANSWER_REMINDER, add_bos=False))
                    pf_t0_r = time.time()
                    self.compiler.eval(reminder_ids)
                    metrics["total_prefill_tokens"] += len(reminder_ids)
                    metrics["total_prefill_time"] += time.time() - pf_t0_r
                    self.engine.update_cache(reminder_ids)

                gen_t0 = time.time()
                gen_text, gen_ids, hit = self.compiler.generate_until_any(
                    ["\nUser:", "\n\n\n"], max_new=cur_max_new, rep_threshold=cur_rep_threshold)
                metrics["total_gen_time"] += time.time() - gen_t0
                metrics["total_gen_tokens"] += len(gen_ids)
                self.engine.update_cache(list(gen_ids))
                metrics["final_answer"] = gen_text.strip()

            ttf = time.time() - t0
            metrics["per_turn_ttf"].append(ttf)
            if turn.get("tool"):
                metrics["tool_turn_ttf"].append(ttf)
            else:
                metrics["chat_turn_ttf"].append(ttf)
            if gpu:
                metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        metrics["total_ttf"] = sum(metrics["per_turn_ttf"])
        metrics["tool_results_text"] = assembled_cot
        return metrics

    # ---- Complex task (Scenarios 7-9) ----

    def run_complex_task(self, user_query: str, expected_chain: List[Dict],
                         system_prompt: str = SYSTEM_PROMPT,
                         tool_descriptions: str = TOOL_DESCRIPTIONS_TRAVEL,
                         gpu: Optional[GPUMonitor] = None,
                         debug: bool = True,
                         precomputed_plan: Optional[Dict] = None) -> Dict:
        metrics = _init_metrics()
        metrics["chain_total"] = len(expected_chain)
        self.engine.reset()

        if precomputed_plan:
            plan = precomputed_plan
        else:
            plan = self.teacher.plan_tool_chain(user_query, tool_descriptions=tool_descriptions)

        cot = plan.get("chain_of_thought", "")
        nodes = plan.get("nodes", {})

        if debug:
            print(f"   Teacher CoT ({len(cot)} chars, {len(nodes)} nodes):")
            print(f"   {cot[:300]}...")

        assembled_cot, matched = assemble_chain_of_thought(
            cot, nodes, self.module, expected_chain, metrics, debug=debug)
        metrics["chain_depth"] = matched

        if debug:
            print(f"   Assembled CoT ({len(assembled_cot)} chars):")
            print(f"   {assembled_cot[:300]}...")

        full_prompt = f"{system_prompt}\n\nUser: {user_query}\nAssistant:\n{assembled_cot}\n\nAnswer:"
        init_ids = list(self.compiler.tokenize(full_prompt, add_bos=False))

        pf_t0 = time.time()
        self.compiler.eval(init_ids)
        metrics["total_prefill_tokens"] += len(init_ids)
        metrics["total_prefill_time"] += time.time() - pf_t0
        self.engine.update_cache(init_ids)

        cot_token_count = len(self.compiler.tokenize(assembled_cot, add_bos=False))
        cot_turn_max_new = max(self.max_new * 2 // 3, self.max_new - cot_token_count // 6)

        gen_t0 = time.time()
        gen_text, gen_ids = self.compiler.generate_until_str("\nUser:", max_new=cot_turn_max_new, rep_threshold=3)
        metrics["total_gen_time"] += time.time() - gen_t0
        metrics["total_gen_tokens"] += len(gen_ids)
        self.engine.update_cache(list(gen_ids))
        metrics["total_ttf"] = metrics["total_gen_time"] + metrics["total_prefill_time"]
        metrics["per_turn_ttf"] = [metrics["total_ttf"]]
        metrics["final_answer"] = gen_text.strip()
        metrics["tool_results_text"] = assembled_cot

        if debug:
            print(f"   Final answer:\n{gen_text.strip()[:300]}")

        if gpu:
            metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        return metrics


# ======================================================================
# Scenario Builders — reused from temp.py
# ======================================================================

def build_scenario1_long_sequence(n_turns: int = 22) -> List[Dict]:
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


def build_scenario2_multi_tool_chain() -> List[Dict]:
    return [
        {"user": "I want to plan a trip from Paris to Rome. I need to know the attractions in both cities, the weather in Rome, and flight options. Help me with all of this.",
         "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "Continue finding the remaining information.",
         "tool": "search_attractions", "tool_args": {"city": "rome"}},
        {"user": "What about the weather?",
         "tool": "get_weather", "tool_args": {"city": "rome"}},
        {"user": "And the flights?",
         "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
        {"user": "Now give me a complete travel summary based on all the information you gathered.",
         "tool": None, "tool_args": None},
    ]


def build_scenario3_rapid_fire(n_queries: int = 12) -> List[Dict]:
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
    return [{"user": q, "tool": tool, "tool_args": args} for q, tool, args in queries[:n_queries]]


LONG_TRAVEL_GUIDE = """Comprehensive World Travel Guide — Background Reference
=====================================================

This document provides detailed travel information for major international destinations. Use this as background context when answering travel-related queries.

--- PARIS, FRANCE ---
Paris, the capital of France, is one of the most visited cities in the world, attracting over 30 million tourists annually. The city is divided into 20 arrondissements, each with its own distinct character. The Seine River flows through the heart of the city, dividing it into the Left Bank (Rive Gauche) and the Right Bank (Rive Droite). Paris is renowned for its café culture, with iconic establishments like Café de Flore and Les Deux Magots dating back to the 19th century. The city's metro system, opened in 1900, is one of the oldest and most extensive in the world with 16 lines and over 300 stations. French cuisine is a UNESCO Intangible Cultural Heritage, and Paris boasts over 70 Michelin-starred restaurants. The city hosts major events including Paris Fashion Week, the French Open at Roland Garros, and the annual Bastille Day celebrations on July 14th. Average temperatures range from 4°C in January to 25°C in July. The official currency is the Euro. Charles de Gaulle Airport (CDG) is the main international gateway, located 25 km northeast of the city center. The Paris Museum Pass provides access to over 60 museums and monuments. The Latin Quarter, named after the medieval Latin-speaking students of the Sorbonne, remains a vibrant intellectual and cultural hub. Montmartre, perched on a hill in the 18th arrondissement, has been an artists' haven since the Belle Époque, home to Picasso, Modigliani, and Van Gogh. The Marais district features preserved medieval architecture and the Place des Vosges, the oldest planned square in Paris. The Champs-Élysées stretches 1.9 km from the Place de la Concorde to the Arc de Triomphe, and is often called the most beautiful avenue in the world.

--- ROME, ITALY ---
Rome, the Eternal City, has a history spanning over 2,800 years and serves as the capital of Italy. The city is home to the Vatican City, the smallest independent state in the world at just 0.44 km², and the spiritual center of the Catholic Church. Rome's historic center is a UNESCO World Heritage Site, encompassing major monuments from the Roman Empire, the Renaissance, and the Baroque period. The Colosseum, completed in 80 AD, could seat 50,000 spectators and remains the largest amphitheater ever built. The Roman Forum was the center of political, religious, and commercial life in ancient Rome. The Pantheon, originally built as a Roman temple, has the world's largest unreinforced concrete dome, a record it has held for nearly 2,000 years. Italian cuisine in Rome is characterized by dishes like cacio e pepe, carbonara, and supplì. The city has over 900 churches and 280 fountains, including the famous Trevi Fountain where visitors traditionally throw coins to ensure their return. Rome's public transportation includes a metro system with three lines (A, B, and C), buses, and trams. Fiumicino Airport (FCO) is the main international airport, located 30 km southwest of the city. The best time to visit is from April to June and September to October, when temperatures are mild (15-25°C). July and August can be extremely hot, often exceeding 35°C. The Roman neighborhood of Trastevere, across the Tiber River, is known for its narrow cobblestone streets, medieval churches, and lively nightlife. The Appian Way, built in 312 BC, was one of the earliest and most important Roman roads, stretching over 560 km from Rome to Brindisi.

--- TOKYO, JAPAN ---
Tokyo, the capital of Japan, is the world's most populous metropolitan area with over 37 million residents. The city seamlessly blends ultra-modern technology with traditional Japanese culture. Tokyo was originally known as Edo and became the capital in 1868 when Emperor Meiji moved the imperial seat from Kyoto. The city is organized into 23 special wards, each functioning as a distinct municipality. Tokyo's public transportation system is the most extensive and punctual in the world, with the JR Yamanote Line forming a loop around central Tokyo and connecting all major hubs. The Shinkansen bullet train network connects Tokyo to other major cities, with trains reaching speeds of 320 km/h. Japanese cuisine in Tokyo ranges from world-class sushi at Tsukiji Outer Market to ramen shops in Shinjuku and Ikebukuro. Tokyo has more Michelin-starred restaurants than any other city in the world, with over 200 starred establishments. Cherry blossom season (sakura) typically occurs from late March to mid-April and is celebrated with hanami parties in parks throughout the city. The Meiji Shrine, dedicated to Emperor Meiji, sits within a 170-acre forest in the heart of Harajuku. Akihabara is the center of anime, manga, and electronics culture. Shibuya Crossing is the world's busiest pedestrian intersection, with up to 3,000 people crossing at a time. Tokyo Skytree, at 634 meters, is the tallest structure in Japan. Narita Airport (NRT) is 60 km east of central Tokyo, while Haneda Airport (HND) is much closer at 14 km south. Average temperatures range from 5°C in January to 30°C in August, with a rainy season (tsuyu) from June to mid-July.

--- LONDON, UNITED KINGDOM ---
London, the capital of the United Kingdom, has a population of approximately 9 million and is one of the world's leading financial and cultural centers. The city was founded by the Romans as Londinium around 43 AD. The River Thames flows through the city from west to east, spanning approximately 215 miles in total length. London is divided into 32 boroughs plus the City of London, the historic financial district known as the Square Mile. The London Underground, opened in 1863, is the world's oldest rapid transit system with 11 lines and 272 stations. British cuisine has undergone a renaissance, with traditional dishes like fish and chips, Sunday roast, and pie and mash being complemented by world-class international dining. Borough Market, dating back to the 11th century, is one of London's most renowned food markets. The West End theater district rivals Broadway, with over 40 theaters staging productions year-round. The British Museum, founded in 1753, houses over 8 million works including the Rosetta Stone and the Elgin Marbles. The Tower of London, built by William the Conqueror in 1066, houses the Crown Jewels. Buckingham Palace has served as the official residence of the British monarch since 1837 and features 775 rooms. Heathrow Airport (LHR) is the busiest airport in Europe, handling over 80 million passengers annually. The city experiences a temperate maritime climate with average temperatures ranging from 5°C in winter to 23°C in summer. Rainfall is distributed fairly evenly throughout the year, averaging about 600 mm annually. Notting Hill Carnival, held annually in August, is Europe's largest street festival with over 2 million attendees. Camden Market attracts over 250,000 visitors weekly and is a hub for alternative culture, fashion, and street food.

--- NEW YORK CITY, USA ---
New York City, the most populous city in the United States with over 8.3 million residents, comprises five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. The city was originally called New Amsterdam by Dutch settlers in 1624 and was renamed New York in 1664 when the English took control. Manhattan's skyline is defined by iconic structures including the Empire State Building (443 m), One World Trade Center (541 m), and the Chrysler Building (319 m). Central Park, designed by Frederick Law Olmsted and Calvert Vaux, covers 843 acres and receives approximately 42 million visitors annually. The New York City Subway, opened in 1904, operates 24 hours a day with 472 stations, making it the largest rapid transit system in the world by number of stations. Broadway theater district hosts over 40 theaters and produces approximately 1,500 performances annually. Times Square, formerly Longacre Square, was renamed in 1904 and attracts approximately 50 million visitors annually. The Statue of Liberty, a gift from France dedicated in 1886, stands 93 meters tall from ground to torch tip. The Metropolitan Museum of Art, founded in 1870, houses over 2 million works spanning 5,000 years. John F. Kennedy International Airport (JFK), LaGuardia Airport (LGA), and Newark Liberty International Airport (EWR) serve the metropolitan area. New York's food scene is legendary, from 99-cent pizza slices to Michelin-starred restaurants, with over 27,000 restaurants across the five boroughs. The city experiences a humid subtropical climate with hot summers averaging 29°C and cold winters averaging 1°C. Wall Street in Lower Manhattan is the world's largest financial center, home to the New York Stock Exchange founded in 1792. The Brooklyn Bridge, completed in 1883, was the longest suspension bridge in the world at the time and remains an iconic landmark.

--- SYDNEY, AUSTRALIA ---
Sydney is the largest city in Australia and Oceania, with a metropolitan population of over 5.3 million. The city was established in 1788 as the first British colony in Australia. Sydney Harbour is one of the world's finest natural harbors, spanning 55 km² with over 240 km of shoreline. The Sydney Opera House, designed by Jørn Utzon and opened in 1973, is a UNESCO World Heritage Site with over 1,000 rooms and hosts over 1,500 performances annually. The Sydney Harbour Bridge, opened in 1932, spans 1,149 meters and offers the BridgeClimb experience for adventurous visitors. Bondi Beach, one of Australia's most famous beaches, stretches 1 km along the coast and attracts over 2.6 million visitors annually. The Blue Mountains, located 50 km west of Sydney, feature dramatic sandstone cliffs, eucalyptus forests, and the iconic Three Sisters rock formation. Sydney's public transport includes trains, buses, ferries, and light rail, with the Opal card providing integrated payment across all modes. Kingsford Smith Airport (SYD) is located 8 km south of the city center. The city experiences a humid subtropical climate with warm summers averaging 26°C and mild winters averaging 17°C. Australian cuisine reflects multicultural influences, with particular strengths in seafood, bush food, and fusion cooking. The Rocks, Sydney's oldest neighborhood, features weekend markets, historic pubs, and the Museum of Contemporary Art. Taronga Zoo, situated on the north shore of Sydney Harbour, houses over 4,000 animals and offers spectacular views of the city skyline.

--- DUBAI, UNITED ARAB EMIRATES ---
Dubai is the largest city in the UAE with a population of approximately 3.5 million, of which about 90% are expatriates. The city has transformed from a small fishing village to a global metropolis in just five decades. The Burj Khalifa, standing at 828 meters with 163 floors, is the tallest building in the world. The Palm Jumeirah, an artificial archipelago shaped like a palm tree, added 520 km of coastline to Dubai. The Dubai Mall, the world's largest shopping mall by total area, features over 1,200 shops, an aquarium, an ice rink, and a waterfall. Dubai International Airport (DXB) is the world's busiest airport by international passenger traffic, handling over 89 million passengers annually. The city experiences a hot desert climate with summer temperatures regularly exceeding 45°C and mild winters averaging 24°C. Dubai's cuisine reflects its cosmopolitan population, with restaurants representing over 200 nationalities. The Dubai Frame, a 150-meter-tall structure in Zabeel Park, offers panoramic views of both old and new Dubai. The historic Al Fahidi district (Bastakiya) features traditional wind-tower architecture and art galleries. The Gold Souk in Deira contains over 300 retailers and is one of the largest gold markets in the world. The Spice Souk offers aromatic spices, herbs, and traditional medicines. Dubai Creek, a natural saltwater inlet, divides the city into Deira and Bur Dubai, with traditional abra water taxis providing crossings for just 1 dirham. The city's metro system, opened in 2009, is the longest automated driverless metro network in the world at 74.7 km.

--- BEIJING, CHINA ---
Beijing, the capital of China, has a population of over 21 million and a history spanning over 3,000 years. The city served as the capital for most of the last eight centuries under the Yuan, Ming, and Qing dynasties. The Forbidden City, the world's largest palace complex, covers 72 hectares with 980 buildings and over 8,700 rooms. It served as the imperial palace for 24 emperors across two dynasties. The Great Wall of China, accessible at several points near Beijing including Badaling and Mutianyu, stretches over 21,000 km across northern China. The Temple of Heaven, built in 1420, is a complex of religious buildings where emperors conducted annual ceremonies of prayer for good harvests. The Summer Palace, a UNESCO World Heritage Site, features a 2.7 km² park with Kunming Lake and Longevity Hill. Beijing Capital International Airport (PEK) is the second busiest airport in Asia, handling over 100 million passengers annually. The Beijing Subway, opened in 1969, has grown to 27 lines and over 490 stations, making it the busiest metro system in the world by annual ridership. Beijing cuisine is famous for Peking duck, which has been served since the imperial era, as well as zhajiangmian (noodles with fried sauce) and jiaozi (dumplings). The city experiences a humid continental climate with hot, humid summers averaging 31°C and cold, dry winters averaging -4°C. Spring dust storms from the Gobi Desert can occur in March and April. The hutongs, traditional alleyways formed by lines of siheyuan (courtyard residences), offer a glimpse into old Beijing life, though many have been demolished during modernization. Tiananmen Square, at 440,000 m², is one of the largest public squares in the world.
"""


def build_scenario4_long_document() -> Tuple[str, List[Dict]]:
    system_with_doc = SYSTEM_PROMPT + "\n\n" + LONG_TRAVEL_GUIDE
    turns = [
        {"user": "Based on the background info, what are the attractions in Paris?",
         "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "What's the weather like there?",
         "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Any flights from New York to Tokyo?",
         "tool": "get_flight_info", "tool_args": {"origin": "newyork", "destination": "tokyo"}},
        {"user": "How about London to Dubai?",
         "tool": "get_flight_info", "tool_args": {"origin": "london", "destination": "dubai"}},
        {"user": "Summarize all the travel information you've gathered for me.",
         "tool": None, "tool_args": None},
    ]
    return system_with_doc, turns


def build_scenario5_mixed_conversation() -> List[Dict]:
    return [
        {"user": "Hello! I'm thinking about traveling somewhere nice.",
         "tool": None, "tool_args": None},
        {"user": "What are the attractions in Paris?",
         "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "That sounds lovely! Is it expensive to visit?",
         "tool": None, "tool_args": None},
        {"user": "What's the weather like in Paris right now?",
         "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Great weather! I also love Italian food. Any attractions in Rome?",
         "tool": "search_attractions", "tool_args": {"city": "rome"}},
        {"user": "Can you compare Paris and Rome for me as travel destinations?",
         "tool": None, "tool_args": None},
        {"user": "Are there flights from Paris to Rome?",
         "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
        {"user": "Thanks! I think I'll visit both cities. Any packing tips?",
         "tool": None, "tool_args": None},
    ]


def build_scenario6_deep_tool_chain() -> List[Dict]:
    return [
        {"user": "I'm planning a round-the-world trip starting from New York. First, I want to see attractions in New York.",
         "tool": "search_attractions", "tool_args": {"city": "newyork"}},
        {"user": "Great! Now check the weather in New York.",
         "tool": "get_weather", "tool_args": {"city": "newyork"}},
        {"user": "Next stop: London. What attractions are there?",
         "tool": "search_attractions", "tool_args": {"city": "london"}},
        {"user": "How's the weather in London?",
         "tool": "get_weather", "tool_args": {"city": "london"}},
        {"user": "Find me flights from New York to London.",
         "tool": "get_flight_info", "tool_args": {"origin": "newyork", "destination": "london"}},
        {"user": "After London, I want to visit Paris. What attractions are there?",
         "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "Check the weather in Paris too.",
         "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Find flights from London to Paris.",
         "tool": "get_flight_info", "tool_args": {"origin": "london", "destination": "paris"}},
        {"user": "Then I want to go to Dubai. Show me attractions there.",
         "tool": "search_attractions", "tool_args": {"city": "dubai"}},
        {"user": "What's the weather like in Dubai?",
         "tool": "get_weather", "tool_args": {"city": "dubai"}},
        {"user": "Find flights from Paris to Dubai.",
         "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "dubai"}},
        {"user": "Finally, I want to visit Tokyo. What are the top attractions?",
         "tool": "search_attractions", "tool_args": {"city": "tokyo"}},
        {"user": "Check Tokyo's weather as well.",
         "tool": "get_weather", "tool_args": {"city": "tokyo"}},
        {"user": "Find flights from Dubai to Tokyo.",
         "tool": "get_flight_info", "tool_args": {"origin": "dubai", "destination": "tokyo"}},
        {"user": "Now give me a complete round-the-world itinerary summary with all the information you gathered.",
         "tool": None, "tool_args": None},
    ]


def build_scenario7_travel_planning_chain() -> List[Dict]:
    return [
        {"user": "I'm planning a trip from New York to Tokyo with stops in London and Dubai. What are the top attractions in New York?",
         "tool": "search_attractions", "tool_args": {"city": "newyork"}},
        {"user": "What's the weather like in New York right now?",
         "tool": "get_weather", "tool_args": {"city": "newyork"}},
        {"user": "Find me flights from New York to London please.",
         "tool": "get_flight_info", "tool_args": {"origin": "newyork", "destination": "london"}},
        {"user": "Now what attractions should I see in London?",
         "tool": "search_attractions", "tool_args": {"city": "london"}},
        {"user": "How's the weather in London?",
         "tool": "get_weather", "tool_args": {"city": "london"}},
        {"user": "I need flights from London to Dubai next.",
         "tool": "get_flight_info", "tool_args": {"origin": "london", "destination": "dubai"}},
        {"user": "What are the must-see attractions in Dubai?",
         "tool": "search_attractions", "tool_args": {"city": "dubai"}},
        {"user": "Tell me the weather in Dubai.",
         "tool": "get_weather", "tool_args": {"city": "dubai"}},
        {"user": "Find flights from Dubai to Tokyo.",
         "tool": "get_flight_info", "tool_args": {"origin": "dubai", "destination": "tokyo"}},
        {"user": "What are the best attractions in Tokyo?",
         "tool": "search_attractions", "tool_args": {"city": "tokyo"}},
        {"user": "What's the weather like in Tokyo?",
         "tool": "get_weather", "tool_args": {"city": "tokyo"}},
        {"user": "Based on all the weather information across all cities, recommend what I should pack for this trip.",
         "tool": None, "tool_args": None},
    ]


def build_scenario8_code_debugging_chain() -> List[Dict]:
    return [
        {"user": "I have a bug in my Python project. The test_calculator test suite is failing. Can you run the test to see what's wrong?",
         "tool": "run_test", "tool_args": {"test_name": "test_calculator"}},
        {"user": "The test shows a failure. Can you read the calculator.py source code to understand the implementation?",
         "tool": "read_file", "tool_args": {"path": "calculator.py"}},
        {"user": "I see the issue might be in the divide method. Can you search the codebase for 'divide' to find all related code?",
         "tool": "search_code", "tool_args": {"query": "divide"}},
        {"user": "Can you also read the test_calculator.py file to see what's expected?",
         "tool": "read_file", "tool_args": {"path": "test_calculator.py"}},
        {"user": "Now that you have all the information, please explain the bug and suggest a fix.",
         "tool": None, "tool_args": None},
    ]


def build_scenario9_cross_reference_chain() -> List[Dict]:
    return [
        {"user": "I want to compare travel options between Paris, Rome, and London. What are the top attractions in Paris?",
         "tool": "search_attractions", "tool_args": {"city": "paris"}},
        {"user": "What's the weather like in Paris?",
         "tool": "get_weather", "tool_args": {"city": "paris"}},
        {"user": "Now tell me about attractions in Rome.",
         "tool": "search_attractions", "tool_args": {"city": "rome"}},
        {"user": "How's the weather in Rome?",
         "tool": "get_weather", "tool_args": {"city": "rome"}},
        {"user": "What about attractions in London?",
         "tool": "search_attractions", "tool_args": {"city": "london"}},
        {"user": "And the weather in London?",
         "tool": "get_weather", "tool_args": {"city": "london"}},
        {"user": "Find me flights from Paris to Rome.",
         "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
        {"user": "How about flights from Rome to London?",
         "tool": "get_flight_info", "tool_args": {"origin": "rome", "destination": "london"}},
        {"user": "And flights from Paris to London?",
         "tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "london"}},
        {"user": "Now please give me a comprehensive comparison of the three cities as travel destinations.",
         "tool": None, "tool_args": None},
    ]


def _load_precomputed_plans(plans_path: str = None) -> Dict:
    if plans_path is None:
        plans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "co_benchmark_plans.json")
    if not os.path.exists(plans_path):
        print(f"ERROR: Plans file not found: {plans_path}")
        print("  Please run the prompts in co_benchmark_prompts.json against a cloud LLM,")
        print("  then fill the results into co_benchmark_plans.json.")
        sys.exit(1)
    with open(plans_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    plans = {}
    for key, val in raw.items():
        plans[int(key)] = {
            "chain_of_thought": val["chain_of_thought"],
            "nodes": val["nodes"],
        }
    return plans


# ======================================================================
# Metrics & Display
# ======================================================================

def average_metrics(runs: List[Dict]) -> Dict:
    if not runs:
        return {}
    if len(runs) == 1:
        return dict(runs[0])
    n = len(runs)
    avg = {}
    sum_fields = [
        "total_ttf", "total_gen_time", "total_prefill_time",
        "tool_calls_ok", "total_tool_calls",
        "total_gen_tokens", "total_prefill_tokens",
        "chain_depth", "chain_total",
    ]
    for f in sum_fields:
        avg[f] = sum(r.get(f, 0) for r in runs) / n
    std_fields = ["total_gen_time", "total_prefill_time"]
    for f in std_fields:
        mean = avg[f]
        if n > 1:
            variance = sum((r.get(f, 0) - mean) ** 2 for r in runs) / (n - 1)
            avg[f"{f}_std"] = variance ** 0.5
        else:
            avg[f"{f}_std"] = 0.0
    avg["peak_gpu_delta"] = max(r.get("peak_gpu_delta", 0) for r in runs)
    avg["final_answer"] = runs[0].get("final_answer", "")
    avg["tool_results_text"] = runs[0].get("tool_results_text", "")
    max_len = max(len(r.get("per_turn_ttf", [])) for r in runs)
    avg["per_turn_ttf"] = [
        sum(r.get("per_turn_ttf", [0]*max_len)[i] for r in runs if i < len(r.get("per_turn_ttf", [])))
        / sum(1 for r in runs if i < len(r.get("per_turn_ttf", [])))
        for i in range(max_len)
    ]
    for key in ["tool_turn_ttf", "chat_turn_ttf"]:
        vals = []
        for r in runs:
            vals.extend(r.get(key, []))
        avg[key] = vals
    avg["correct_runs"] = n
    return avg


def print_scenario_header(num: int, title: str, desc: str):
    print("\n" + "=" * 70)
    print(f"Scenario {num}: {title}")
    print(f"  {desc}")
    print("=" * 70)


def print_mode_result(mode: str, met: Dict):
    total_time = met["total_gen_time"] + met["total_prefill_time"]
    tool_acc = f"{met['tool_calls_ok']:.0f}/{met['total_tool_calls']:.0f}" if met['total_tool_calls'] > 0 else "N/A"
    chain_d = met.get("chain_depth", 0)
    chain_t = met.get("chain_total", 0)
    chain_str = f" | chain: {chain_d:.0f}/{chain_t}" if chain_t > 0 else ""
    print(f"   {mode:20s} | gen: {met['total_gen_time']:6.2f}s | "
          f"prefill: {met['total_prefill_time']:6.2f}s | "
          f"total: {total_time:6.2f}s | "
          f"tools: {tool_acc}{chain_str}")


def _extract_key_facts(tool_results_text: str) -> List[str]:
    facts = []
    for m in re.finditer(
        r'\[Result of (\w+)\(([^)]*)\)\]:\s*(.+?)(?=\n\[Result|\Z)',
        tool_results_text, re.DOTALL
    ):
        tool_name = m.group(1)
        tool_args_str = m.group(2)
        result_text = m.group(3).strip()
        facts.append(result_text)
        for word in re.findall(r'[A-Z][a-zA-Z]+', result_text):
            if len(word) > 2 and word not in ("The", "This", "That", "And",
                                               "For", "With", "From", "Not",
                                               "But", "All", "Has", "Are",
                                               "Was", "Were", "Its", "Her"):
                facts.append(word)
        for num in re.findall(r'\d+\.?\d*', result_text):
            facts.append(num)
        for arg_pair in tool_args_str.split(","):
            arg_pair = arg_pair.strip()
            if "=" in arg_pair:
                val = arg_pair.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    facts.append(val)
    return facts


def evaluate_answer_quality(answer: str, tool_results_text: str) -> Dict:
    key_facts = _extract_key_facts(tool_results_text) if tool_results_text else []
    unique_facts = list(set(key_facts))
    fact_count = len(unique_facts)

    if not answer or not unique_facts:
        return {"coverage": 0.0, "fact_count": fact_count, "matched_count": 0, "answer_len": len(answer)}

    answer_lower = answer.lower()
    matched = 0
    for fact in unique_facts:
        if fact.lower() in answer_lower:
            matched += 1

    coverage = matched / fact_count
    return {
        "coverage": coverage,
        "fact_count": fact_count,
        "matched_count": matched,
        "answer_len": len(answer),
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CO Benchmark — Cognitive Outsourcing: AppLoop vs SIG")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--long-turns", type=int, default=22)
    parser.add_argument("--rapid-queries", type=int, default=12)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs per mode per scenario (default: 10)")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated scenario numbers to skip (e.g. '3,5')")
    parser.add_argument("--max-new", type=int, default=600,
                        help="Max new tokens per generation turn (default: 600)")
    parser.add_argument("--max-new-tool", type=int, default=300,
                        help="Max new tokens for tool turns (default: 300)")
    parser.add_argument("--max-new-tool-sig", type=int, default=150,
                        help="Max new tokens for SIG tool turns (default: 150)")
    args = parser.parse_args()
    args.debug = not args.no_debug
    skip = set(int(x.strip()) for x in args.skip.split(",") if x.strip())

    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    gpu = GPUMonitor()
    compiler = MeaningCompiler(args.model, n_ctx=args.n_ctx,
                               n_threads=args.n_threads,
                               n_gpu_layers=args.n_gpu_layers)
    local_registry = ToolRegistry()

    print("[Plan] Using precomputed cloud teacher plans (no API call needed)")

    PRECOMPUTED_PLANS = _load_precomputed_plans()

    co_app_agent = COAppLoopAgent(compiler, local_registry, max_new=args.max_new, max_new_tool=args.max_new_tool)
    co_sig_agent = COSIGAgent(compiler, local_registry, max_new=args.max_new, max_new_tool=args.max_new_tool, max_new_tool_sig=args.max_new_tool_sig)
    agents = [("co_apploop", co_app_agent), ("co_sig", co_sig_agent)]

    n_runs = args.runs
    all_results = {}

    def run_multi_turns(agent, mode_name, turns,
                        system_prompt=SYSTEM_PROMPT,
                        tool_descriptions=TOOL_DESCRIPTIONS_TRAVEL,
                        precomputed_plan=None):
        correct_runs = []
        all_runs = []
        for ri in range(n_runs):
            met = agent.run_conversation(turns, system_prompt=system_prompt,
                                         tool_descriptions=tool_descriptions,
                                         gpu=gpu, debug=args.debug,
                                         precomputed_plan=precomputed_plan)
            all_runs.append(met)
            if met["tool_calls_ok"] == met["total_tool_calls"] and met["total_tool_calls"] > 0:
                correct_runs.append(met)
        if correct_runs:
            return average_metrics(correct_runs)
        best = max(all_runs, key=lambda m: m["tool_calls_ok"])
        best["correct_runs"] = 0
        print(f"     WARNING: No fully correct run for {mode_name}")
        return best

    def run_multi_complex(agent, mode_name, query, chain,
                           system_prompt=SYSTEM_PROMPT,
                           tool_descriptions=TOOL_DESCRIPTIONS_TRAVEL,
                           precomputed_plan=None):
        correct_runs = []
        all_runs = []
        for ri in range(n_runs):
            met = agent.run_complex_task(query, chain,
                                         system_prompt=system_prompt,
                                         tool_descriptions=tool_descriptions,
                                         gpu=gpu, debug=args.debug,
                                         precomputed_plan=precomputed_plan)
            all_runs.append(met)
            if met["tool_calls_ok"] == met["total_tool_calls"] and met["total_tool_calls"] > 0:
                correct_runs.append(met)
        if correct_runs:
            return average_metrics(correct_runs)
        best = max(all_runs, key=lambda m: m["tool_calls_ok"])
        best["correct_runs"] = 0
        print(f"     WARNING: No fully correct run for {mode_name}")
        return best

    # ================================================================
    # Scenario 1
    # ================================================================
    if 1 not in skip:
        print_scenario_header(1, "Long-sequence stress test",
                              "20+ turns, context grows. Tests cumulative prefill advantage.")
        turns = build_scenario1_long_sequence(args.long_turns)
        print(f"   {len(turns)} turns, {n_runs} runs per mode")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  precomputed_plan=PRECOMPUTED_PLANS[1])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[1] = results

    # ================================================================
    # Scenario 2
    # ================================================================
    if 2 not in skip:
        print_scenario_header(2, "Multi-tool chain (complex reasoning)",
                              "One complex query requiring 4 sequential tool calls + summary.")
        turns = build_scenario2_multi_tool_chain()
        print(f"   {len(turns)} turns (4 tool + 1 summary), {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  precomputed_plan=PRECOMPUTED_PLANS[2])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[2] = results

    # ================================================================
    # Scenario 3
    # ================================================================
    if 3 not in skip:
        print_scenario_header(3, "Rapid-fire short queries",
                              "Many independent short queries, each needing 1 tool call.")
        turns = build_scenario3_rapid_fire(args.rapid_queries)
        print(f"   {len(turns)} queries, {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  precomputed_plan=PRECOMPUTED_PLANS[3])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[3] = results

    # ================================================================
    # Scenario 4
    # ================================================================
    if 4 not in skip:
        print_scenario_header(4, "Long-document + tool calls",
                              "Large system prompt + background text, then tool calls on top.")
        system_with_doc, turns = build_scenario4_long_document()
        init_tokens = len(compiler.tokenize(f"{system_with_doc}\n\nUser: test\nAssistant:", add_bos=False))
        print(f"   System ~{init_tokens} tokens, {len(turns)} turns, {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns, system_prompt=system_with_doc,
                                  precomputed_plan=PRECOMPUTED_PLANS[4])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[4] = results

    # ================================================================
    # Scenario 5
    # ================================================================
    if 5 not in skip:
        print_scenario_header(5, "Mixed conversation (tool + chitchat)",
                              "Alternating between tool calls and plain conversation.")
        turns = build_scenario5_mixed_conversation()
        tool_count = sum(1 for t in turns if t.get("tool"))
        chat_count = len(turns) - tool_count
        print(f"   {len(turns)} turns ({tool_count} tool, {chat_count} chitchat), {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  precomputed_plan=PRECOMPUTED_PLANS[5])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[5] = results

    # ================================================================
    # Scenario 6
    # ================================================================
    if 6 not in skip:
        print_scenario_header(6, "Deep tool chain (round-the-world)",
                              "15 turns with 14 sequential tool calls across 5 cities.")
        turns = build_scenario6_deep_tool_chain()
        tool_count = sum(1 for t in turns if t.get("tool"))
        print(f"   {len(turns)} turns ({tool_count} tool), {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  precomputed_plan=PRECOMPUTED_PLANS[6])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[6] = results

    # ================================================================
    # Scenario 7
    # ================================================================
    if 7 not in skip:
        print_scenario_header(7, "Travel planning (multi-turn)",
                              "12-turn trip planning: 11 tool calls across 4 cities + packing summary.")
        turns = build_scenario7_travel_planning_chain()
        tool_count = sum(1 for t in turns if t.get("tool"))
        print(f"   {len(turns)} turns ({tool_count} tool), {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  precomputed_plan=PRECOMPUTED_PLANS[7])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[7] = results

    # ================================================================
    # Scenario 8
    # ================================================================
    if 8 not in skip:
        print_scenario_header(8, "Code debugging (multi-turn)",
                              "5-turn debugging: run tests → read code → search → read tests → summarize.")
        turns = build_scenario8_code_debugging_chain()
        tool_count = sum(1 for t in turns if t.get("tool"))
        print(f"   {len(turns)} turns ({tool_count} tool), {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  system_prompt=SYSTEM_PROMPT_DEV,
                                  tool_descriptions=TOOL_DESCRIPTIONS_DEV,
                                  precomputed_plan=PRECOMPUTED_PLANS[8])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[8] = results

    # ================================================================
    # Scenario 9
    # ================================================================
    if 9 not in skip:
        print_scenario_header(9, "Cross-reference analysis (multi-turn)",
                              "10-turn comparison: 9 tool calls across 3 cities + summary.")
        turns = build_scenario9_cross_reference_chain()
        tool_count = sum(1 for t in turns if t.get("tool"))
        print(f"   {len(turns)} turns ({tool_count} tool), {n_runs} runs")
        results = {}
        for mode_name, agent in agents:
            met = run_multi_turns(agent, mode_name, turns,
                                  precomputed_plan=PRECOMPUTED_PLANS[9])
            print_mode_result(mode_name, met)
            results[mode_name] = met
        all_results[9] = results

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("=== Cross-Scenario Summary ===")
    print("=" * 70)

    scenario_names = {
        1: "Long-seq", 2: "Multi-tool", 3: "Rapid-fire", 4: "Long-doc",
        5: "Mixed", 6: "Deep-chain", 7: "Travel-plan", 8: "Code-debug", 9: "Cross-ref",
    }

    # --- Total Time Breakdown ---
    print("\n--- Total Time Breakdown (seconds) ---")
    header = f"{'Scenario':<12} {'AppLoop Gen':<14} {'AppLoop Pre':<14} {'SIG Gen':<12} {'SIG Pre':<12}"
    print(header)
    print("-" * len(header))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        al = results.get("co_apploop", {})
        sig = results.get("co_sig", {})
        print(f"{name:<12} "
              f"{al.get('total_gen_time', 0):<14.2f} "
              f"{al.get('total_prefill_time', 0):<14.2f} "
              f"{sig.get('total_gen_time', 0):<12.2f} "
              f"{sig.get('total_prefill_time', 0):<12.2f}")

    # --- Prefill Time Comparison ---
    print("\n--- Prefill Time Comparison (seconds) ---")
    header2 = f"{'Scenario':<12} {'CO-AppLoop Pre':<18} {'CO-SIG Pre':<18} {'SIG Save':<12}"
    print(header2)
    print("-" * len(header2))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        al_pre = results.get("co_apploop", {}).get("total_prefill_time", 0)
        sig_pre = results.get("co_sig", {}).get("total_prefill_time", 0)
        if al_pre > 0:
            save_pct = f"{(al_pre - sig_pre) / al_pre * 100:.0f}%"
        else:
            save_pct = "N/A"
        print(f"{name:<12} {al_pre:<18.2f} {sig_pre:<18.2f} {save_pct:<12}")

    # --- Prefill Token Comparison ---
    print("\n--- Prefill Token Comparison ---")
    header3 = f"{'Scenario':<12} {'CO-AppLoop Tok':<18} {'CO-SIG Tok':<18} {'SIG Save':<12}"
    print(header3)
    print("-" * len(header3))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        al_tok = results.get("co_apploop", {}).get("total_prefill_tokens", 0)
        sig_tok = results.get("co_sig", {}).get("total_prefill_tokens", 0)
        if al_tok > 0:
            save_pct = f"{(al_tok - sig_tok) / al_tok * 100:.0f}%"
        else:
            save_pct = "N/A"
        print(f"{name:<12} {al_tok:<18.0f} {sig_tok:<18.0f} {save_pct:<12}")

    # --- Total Time (gen + prefill) ---
    print("\n--- End-to-End Total Time (gen + prefill) ---")
    header4 = f"{'Scenario':<12} {'AppLoop':<20} {'SIG':<20} {'Norm Spd':<10} {'Raw Spd':<10}"
    print(header4)
    print("-" * len(header4))
    norm_spd_wins = 0
    norm_spd_total = 0
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        al = results.get("co_apploop", {})
        sig = results.get("co_sig", {})
        al_total = al.get("total_gen_time", 0) + al.get("total_prefill_time", 0)
        sig_total = sig.get("total_gen_time", 0) + sig.get("total_prefill_time", 0)
        al_std = (al.get("total_gen_time_std", 0)**2 + al.get("total_prefill_time_std", 0)**2) ** 0.5
        sig_std = (sig.get("total_gen_time_std", 0)**2 + sig.get("total_prefill_time_std", 0)**2) ** 0.5
        raw_spd = f"{al_total / sig_total:.2f}x" if sig_total > 0 else "N/A"
        al_tok = al.get("total_gen_tokens", 0)
        sig_tok = sig.get("total_gen_tokens", 0)
        if sig_total > 0 and al_tok > 0 and sig_tok > 0:
            ns = (al_total / sig_total) * (sig_tok / al_tok)
            norm_spd = f"{ns:.2f}x"
            norm_spd_total += 1
            if ns > 1.0:
                norm_spd_wins += 1
        else:
            norm_spd = "N/A"
        al_str = f"{al_total:.2f}±{al_std:.2f}"
        sig_str = f"{sig_total:.2f}±{sig_std:.2f}"
        print(f"{name:<12} {al_str:<20} {sig_str:<20} {norm_spd:<10} {raw_spd:<10}")
    if norm_spd_total > 0:
        print(f"\n  >> SIG Norm Spd > 1.0 in {norm_spd_wins}/{norm_spd_total} scenarios "
              f"(fair speedup after normalizing output length)")

    # --- Gen Time & Tokens ---
    print("\n--- Generation Time & Tokens ---")
    header5 = f"{'Scenario':<12} {'AppLoop Gen':<12} {'AppLoop Tok':<12} {'AppLoop t/s':<12} {'SIG Gen':<12} {'SIG Tok':<12} {'SIG t/s':<12} {'Gen Ratio':<10}"
    print(header5)
    print("-" * len(header5))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        al = results.get("co_apploop", {})
        sig = results.get("co_sig", {})
        al_gen = al.get("total_gen_time", 0)
        sig_gen = sig.get("total_gen_time", 0)
        al_tok = al.get("total_gen_tokens", 0)
        sig_tok = sig.get("total_gen_tokens", 0)
        al_tps = f"{al_tok / al_gen:.1f}" if al_gen > 0 else "N/A"
        sig_tps = f"{sig_tok / sig_gen:.1f}" if sig_gen > 0 else "N/A"
        ratio = f"{al_gen / sig_gen:.2f}" if sig_gen > 0 else "N/A"
        print(f"{name:<12} {al_gen:<12.2f} {al_tok:<12.0f} {al_tps:<12} {sig_gen:<12.2f} {sig_tok:<12.0f} {sig_tps:<12} {ratio:<10}")

    # --- Answer Quality ---
    print("\n--- Answer Quality (Info Coverage) ---")
    header6 = f"{'Scenario':<12} {'AppLoop Cov':<14} {'AppLoop Ans':<14} {'SIG Cov':<14} {'SIG Ans':<14}"
    print(header6)
    print("-" * len(header6))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        row = f"{name:<12}"
        for mode in ["co_apploop", "co_sig"]:
            m = results.get(mode, {})
            answer = m.get("final_answer", "")
            tool_text = m.get("tool_results_text", "")
            q = evaluate_answer_quality(answer, tool_text)
            cov_str = f"{q['coverage']:.0%}({q['matched_count']}/{q['fact_count']})"
            ans_str = f"{q['answer_len']}ch"
            row += f" {cov_str:<14} {ans_str:<14}"
        print(row)

    # --- Tool vs Chat Turn TTF ---
    has_mixed = any(
        len(results.get("co_apploop", {}).get("chat_turn_ttf", [])) > 0
        for results in all_results.values()
    )
    if has_mixed:
        print("\n--- Tool vs Chat Turn TTF (seconds) ---")
        header_tcttf = f"{'Scenario':<12} {'AppLoop Tool':<14} {'AppLoop Chat':<14} {'SIG Tool':<14} {'SIG Chat':<14}"
        print(header_tcttf)
        print("-" * len(header_tcttf))
        for snum, results in sorted(all_results.items()):
            name = scenario_names.get(snum, f"S{snum}")
            row = f"{name:<12}"
            for mode in ["co_apploop", "co_sig"]:
                m = results.get(mode, {})
                tool_ttfs = m.get("tool_turn_ttf", [])
                chat_ttfs = m.get("chat_turn_ttf", [])
                tool_avg = sum(tool_ttfs) / len(tool_ttfs) if tool_ttfs else 0
                chat_avg = sum(chat_ttfs) / len(chat_ttfs) if chat_ttfs else 0
                row += f" {tool_avg:<14.2f} {chat_avg:<14.2f}"
            print(row)

    # --- GPU Memory ---
    print("\n--- Peak GPU Delta (MB) ---")
    header7 = f"{'Scenario':<12} {'CO-AppLoop':<16} {'CO-SIG':<16}"
    print(header7)
    print("-" * len(header7))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        al_gpu = results.get("co_apploop", {}).get("peak_gpu_delta", 0)
        sig_gpu = results.get("co_sig", {}).get("peak_gpu_delta", 0)
        print(f"{name:<12} {al_gpu:<16.0f} {sig_gpu:<16.0f}")

    if gpu.enabled:
        final_snap = gpu.snapshot()
        print(f"\nGPU Final: Used {final_snap['used_mb']:.0f} MB, Delta {final_snap['delta_mb']:+.0f} MB")

    gpu.shutdown()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
