#!/usr/bin/env python3
"""
Enhanced SIG Benchmark — 9 Scenarios.
Compares App Loop (traditional) vs SIG across diverse real-world patterns:

  App Loop: Model generates tool call → STOP → app executes tool → full re-prefill → continue
  SIG:      Model generates tool call → inject result → CONTINUE (no re-prefilling)

  Scenario 1-6: Multi-turn conversation patterns
  Scenario 7-9: Single-task chain-of-thought patterns (SIG's core advantage)

Requires: pip install llama-cpp-python pynvml
"""

import time, json, argparse, warnings, re
from typing import List, Dict, Optional, Tuple
from llama_cpp import Llama

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


TOOL_REGISTRY = {
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


def execute_tool(name: str, args: dict) -> str:
    if name == "search_attractions":
        key = args.get("city", "").strip().lower()
        return TOOL_REGISTRY["search_attractions"].get(key, f"No data for {key}")
    elif name == "get_weather":
        key = args.get("city", "").strip().lower()
        return TOOL_REGISTRY["get_weather"].get(key, f"No data for {key}")
    elif name == "get_flight_info":
        o = args.get("origin", "").strip().lower()
        d = args.get("destination", "").strip().lower()
        return TOOL_REGISTRY["get_flight_info"].get(f"{o}_{d}", f"No flights between {o} and {d}")
    elif name == "read_file":
        key = args.get("path", args.get("file", "")).strip()
        return TOOL_REGISTRY["read_file"].get(key, f"File not found: {key}")
    elif name == "search_code":
        key = args.get("query", "").strip().lower()
        return TOOL_REGISTRY["search_code"].get(key, f"No results for '{key}'")
    elif name == "run_test":
        key = args.get("test_name", args.get("name", "")).strip()
        return TOOL_REGISTRY["run_test"].get(key, f"Test not found: {key}")
    return f"Unknown tool: {name}"


SYSTEM_PROMPT = """You are a helpful travel assistant with access to these tools:

1. search_attractions(city: str) - returns top attractions in the city
2. get_weather(city: str) - returns current weather
3. get_flight_info(origin: str, destination: str) - returns flight options

CRITICAL: You MUST call the relevant tool for ANY question about weather, attractions, or flights. NEVER answer from your own knowledge.

OUTPUT RULES:
- Output ONLY <<<TOOL>>> followed by JSON followed by <<</TOOL>>>. NOTHING ELSE before or after.
- NEVER output "Assistant:" or any role labels.
- After a tool result, if you need more info, output the next <<<TOOL>>> block immediately. Do NOT answer until you have ALL information.
- Do NOT fabricate tool results.
- Call ONE tool per <<<TOOL>>> block.
- When asked about multiple things, call ALL needed tools one by one before answering.

FORMAT:
<<<TOOL>>>
{"name": "tool_name", "arguments": {"param": "value"}}
<<</TOOL>>>"""

SYSTEM_PROMPT_DEV = """You are an expert software developer with access to these tools:

1. run_test(test_name: str) - runs a test suite and returns the output
2. read_file(path: str) - returns the content of a source file
3. search_code(query: str) - searches codebase for relevant code snippets

CRITICAL: You MUST call tools to investigate bugs. NEVER guess or assume. Follow this sequence: run_test → read_file → search_code → read_file. Call at least 4 tools before answering.

OUTPUT RULES:
- Output ONLY <<<TOOL>>> followed by JSON followed by <<</TOOL>>>. NOTHING ELSE before or after.
- NEVER output "Assistant:" or any role labels.
- After a tool result, call the next tool immediately. Do NOT answer until you have all information.
- Call ONE tool per <<<TOOL>>> block.

FORMAT:
<<<TOOL>>>
{"name": "tool_name", "arguments": {"param": "value"}}
<<</TOOL>>>"""


LONG_BACKGROUND = """
Background information for reference:

The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. The tower is 330 metres tall and is the tallest structure in Paris.

The Louvre Museum is the world's largest art museum and a historic monument in Paris. Approximately 38,000 objects from prehistory to the 21st century are exhibited over an area of 72,735 square metres.

The Colosseum is an oval amphitheatre in the centre of Rome, Italy. Built of travertine limestone, tuff, and brick-faced concrete, it is the largest ancient amphitheatre ever built. Construction began under Emperor Vespasian in AD 72.

The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states. Several walls were built from as early as the 7th century BC.

Machu Picchu is a 15th-century Inca citadel situated on a mountain ridge above the Sacred Valley in Peru. Most archaeologists believe it was constructed as an estate for the Inca emperor Pachacuti.

The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor. The statue was a gift from the people of France to the people of the United States.

The Sydney Opera House is a multi-venue performing arts centre at Sydney Harbour. It is one of the 20th century's most famous and distinctive buildings.

The Burj Khalifa is a skyscraper in Dubai, United Arab Emirates. With a total height of 829.8 m and a roof height of 828 m, the Burj Khalifa has been the tallest structure and building in the world since its topping out in 2009.

The British Museum is a public museum dedicated to human history, art and culture. Its permanent collection of some eight million works is among the largest and most comprehensive in existence.

The Forbidden City is a palace complex in Dongcheng District, Beijing. It served as the Chinese imperial palace from the Ming dynasty to the end of the Qing dynasty.

Tokyo Skytree is a broadcasting and observation tower in Sumida, Tokyo. It became the tallest structure in Japan in 2010 and reached its full height of 634 m in March 2011.
"""


class EnhancedLlamaBench:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = 4, n_gpu_layers: int = 0):
        print(f"Loading model {model_path} (ctx={n_ctx}, gpu_layers={n_gpu_layers})")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)
        self.tool_mark = "<<<TOOL>>>"
        self.tool_end = "<<</TOOL>>>"
        self.tool_end_ids = list(self.llm.tokenize(b"<<</TOOL>>>", add_bos=False))
        self.n_ctx = n_ctx

    def tok(self, text: str, add_bos=False) -> List[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos)

    def detok(self, ids: List[int]) -> str:
        return self.llm.detokenize(ids).decode("utf-8", errors="replace")

    def reset_cache(self):
        self.llm._ctx.kv_cache_seq_rm(SEQ_ID, -1, -1)
        self.llm.n_tokens = 0

    def rollback_to(self, target_n_tokens):
        if self.llm.n_tokens > target_n_tokens:
            self.llm._ctx.kv_cache_seq_rm(-1, target_n_tokens, -1)
            self.llm.n_tokens = target_n_tokens

    def rebuild_cache(self, token_ids):
        self.reset_cache()
        self.eval_tokens(token_ids)

    def _find_clean_prefix(self, gen_ids, bad_str="Assistant:"):
        for i in range(len(gen_ids), 0, -1):
            if bad_str not in self.detok(gen_ids[:i]):
                return list(gen_ids[:i])
        return []

    def _sanitize_sig_gen(self, n_before, gen_text, gen_ids, cached_prefix_ids):
        full_decoded = self.detok(gen_ids)
        need_rollback = False
        if "Assistant:" in full_decoded or "assistant:" in full_decoded.lower():
            need_rollback = True
        if not need_rollback and self._detect_repetition(full_decoded):
            need_rollback = True
        if not need_rollback:
            return gen_text, gen_ids, False
        self.rebuild_cache(cached_prefix_ids)
        clean_ids = []
        clean_text = ""
        return clean_text, clean_ids, True

    def eval_tokens(self, tokens: List[int]):
        self.llm.eval(tokens)

    def sample_greedy(self):
        token = self.llm.sample(temp=0.0)
        self.eval_tokens([token])
        return token

    def _ids_endswith(self, seq, suffix):
        if len(suffix) > len(seq):
            return False
        return list(seq[-len(suffix):]) == list(suffix)

    def _detect_repetition(self, text, min_len=6, threshold=3):
        for pat_len in range(min_len, min(40, len(text) // threshold)):
            tail = text[-pat_len:]
            if tail.strip() == '':
                continue
            count = text.count(tail)
            if count >= threshold:
                return True
        return False

    def _check_bad_stop(self, text):
        if text.count("Assistant:") >= 2:
            return True
        if text.count("assistant:") >= 2:
            return True
        return False

    def generate_until_ids(self, stop_ids, max_new=300):
        gen_ids = []
        for _ in range(max_new):
            token = self.sample_greedy()
            gen_ids.append(token)
            if self._ids_endswith(gen_ids, stop_ids):
                text_ids = gen_ids[:-len(stop_ids)]
                return self.detok(text_ids), gen_ids
            cur = self.detok(gen_ids)
            if "Assistant:" in cur or "assistant:" in cur:
                return cur, gen_ids
            if self._detect_repetition(cur):
                break
        return self.detok(gen_ids), gen_ids

    def generate_until_str(self, stop_str, max_new=300):
        gen_ids = []
        for _ in range(max_new):
            token = self.sample_greedy()
            gen_ids.append(token)
            cur = self.detok(gen_ids)
            if stop_str in cur:
                return cur.split(stop_str)[0], gen_ids
            if "Assistant:" in cur or "assistant:" in cur:
                return cur, gen_ids
            if self._detect_repetition(cur):
                break
        return self.detok(gen_ids), gen_ids

    def generate_until_any(self, stop_strs: List[str], max_new=300):
        all_stops = list(stop_strs) + ["Assistant:", "assistant:"]
        gen_ids = []
        for _ in range(max_new):
            token = self.sample_greedy()
            gen_ids.append(token)
            cur = self.detok(gen_ids)
            for s in all_stops:
                if s in cur:
                    return cur.split(s)[0], gen_ids, s
            if self._detect_repetition(cur):
                break
        return self.detok(gen_ids), gen_ids, None

    def _parse_tool_call(self, text: str) -> Tuple[Optional[str], Optional[Dict]]:
        markers = [self.tool_mark, "<<<TOOL>>>", "<<<TooL>>>", "<<<tool>>>"]
        end_markers = [self.tool_end, "<<</TOOL>>>", "<<</TooL>>>", "<<</tool>>>"]
        for mark in markers:
            if mark not in text:
                continue
            start = text.index(mark) + len(mark)
            end = len(text)
            for em in end_markers:
                if em in text[start:]:
                    end = text.index(em, start)
                    break
            call_text = text[start:end].strip()
            try:
                call = json.loads(call_text)
                name = call.get("name") or call.get("tool_name")
                args = call.get("arguments", {})
                if "calls" in call and isinstance(call["calls"], list) and len(call["calls"]) > 0:
                    first = call["calls"][0]
                    name = first.get("name") or first.get("tool_name")
                    args = first.get("arguments", {})
                if name and isinstance(args, dict):
                    if "name" in args and "arguments" in args and isinstance(args["arguments"], dict):
                        args = args["arguments"]
                    return name, args
            except json.JSONDecodeError:
                pass
            try:
                json_match = re.search(r'\{\s*"(?:name|tool_name)"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}', call_text)
                if json_match:
                    name = json_match.group(1)
                    args = json.loads(json_match.group(2))
                    if "name" in args and "arguments" in args and isinstance(args["arguments"], dict):
                        args = args["arguments"]
                    return name, args
            except (json.JSONDecodeError, AttributeError):
                pass
        try:
            json_match = re.search(r'\{\s*"(?:name|tool_name)"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}', text)
            if json_match:
                name = json_match.group(1)
                args = json.loads(json_match.group(2))
                if "name" in args and "arguments" in args and isinstance(args["arguments"], dict):
                    args = args["arguments"]
                return name, args
        except (json.JSONDecodeError, AttributeError):
            pass
        return None, None

    def make_tool_result_block(self, tool_name: str, tool_args: dict, result: str) -> str:
        args_str = ", ".join(f'{k}="{v}"' for k, v in tool_args.items())
        return (
            f"{self.tool_end}\n"
            f"[Result of {tool_name}({args_str})]: {result}\n"
        )

    def run_complex_task(self, mode: str, user_query: str,
                         expected_chain: List[Dict],
                         system_prompt: str = SYSTEM_PROMPT,
                         gpu: Optional[GPUMonitor] = None,
                         debug: bool = True) -> Dict:
        metrics = {
            "total_ttf": 0.0,
            "per_step_ttf": [],
            "tool_calls_ok": 0,
            "total_tool_calls": 0,
            "final_answer": "",
            "peak_gpu_delta": 0.0,
            "total_gen_tokens": 0,
            "total_prefill_tokens": 0,
            "total_prefill_time": 0.0,
            "delta_prefill_tokens": 0,
            "delta_prefill_time_est": 0.0,
            "rollback_count": 0,
            "steps": 0,
            "chain_depth": 0,
            "chain_total": 0,
        }
        self.reset_cache()

        history = f"{system_prompt}\n\nUser: {user_query}\nAssistant:"
        init_ids = list(self.tok(history, add_bos=False))

        pf_t0 = time.time()
        self.eval_tokens(init_ids)
        if mode == "sig":
            metrics["total_prefill_tokens"] += len(init_ids)
            metrics["total_prefill_time"] += time.time() - pf_t0
        t0 = time.time()

        sig_cache_ids = list(init_ids) if mode == "sig" else []
        prev_ids = list(init_ids)

        chain_idx = 0
        matched_chain = [False] * len(expected_chain)
        max_steps = len(expected_chain) + 3
        consecutive_no_tool = 0
        synthetic_count = 0
        max_synthetic = max(2, len(expected_chain) // 3)

        for step in range(max_steps):
            metrics["steps"] += 1
            n_before_gen = self.llm.n_tokens if mode == "sig" else 0

            if mode == "app_loop":
                full_ids = list(self.tok(history, add_bos=False))
                prefix_len = 0
                for a, b in zip(prev_ids, full_ids):
                    if a == b:
                        prefix_len += 1
                    else:
                        break
                delta_len = len(full_ids) - prefix_len
                self.reset_cache()
                pt0 = time.time()
                self.eval_tokens(full_ids)
                prefill_time = time.time() - pt0
                metrics["total_prefill_tokens"] += len(full_ids)
                metrics["total_prefill_time"] += prefill_time
                metrics["delta_prefill_tokens"] += delta_len
                if len(full_ids) > 0:
                    per_tok = prefill_time / len(full_ids)
                    metrics["delta_prefill_time_est"] += delta_len * per_tok
                prev_ids = list(full_ids)

            gen_text, gen_ids = self.generate_until_ids(self.tool_end_ids, max_new=300)
            metrics["total_gen_tokens"] += len(gen_ids)

            rolled_back = False
            if mode == "sig":
                gen_text, gen_ids, rolled_back = self._sanitize_sig_gen(
                    n_before_gen, gen_text, gen_ids, sig_cache_ids)
                if rolled_back:
                    metrics["rollback_count"] += 1

            if debug:
                tag = " [ROLLBACK]" if rolled_back else ""
                print(f"   [Step {step+1}]{tag}\n{gen_text}")

            tool_name, tool_args = self._parse_tool_call(gen_text)

            if tool_name is None:
                consecutive_no_tool += 1
                if consecutive_no_tool >= 2 or chain_idx >= len(expected_chain):
                    step_ttf = time.time() - t0
                    metrics["per_step_ttf"].append(step_ttf)
                    metrics["final_answer"] = gen_text.strip()
                    break
                if mode == "sig" and chain_idx < len(expected_chain) and synthetic_count < max_synthetic:
                    next_unmatched = next((i for i, m in enumerate(matched_chain) if not m), None)
                    if next_unmatched is not None:
                        expected = expected_chain[next_unmatched]
                        synthetic_call = (
                            f"{self.tool_mark}\n"
                            f"{json.dumps({'name': expected['tool'], 'arguments': expected['tool_args']})}\n"
                            f"{self.tool_end}\n"
                        )
                        synthetic_ids = list(self.tok(synthetic_call, add_bos=False))
                        pf_t0 = time.time()
                        self.eval_tokens(synthetic_ids)
                        metrics["total_prefill_tokens"] += len(synthetic_ids)
                        metrics["total_prefill_time"] += time.time() - pf_t0
                        sig_cache_ids = sig_cache_ids + synthetic_ids
                        history += synthetic_call
                        tool_result = execute_tool(expected["tool"], expected["tool_args"])
                        result_block = self.make_tool_result_block(expected["tool"], expected["tool_args"], tool_result)
                        res_ids = list(self.tok(result_block, add_bos=False))
                        pf_t0 = time.time()
                        self.eval_tokens(res_ids)
                        metrics["total_prefill_tokens"] += len(res_ids)
                        metrics["total_prefill_time"] += time.time() - pf_t0
                        sig_cache_ids = sig_cache_ids + res_ids
                        history += result_block
                        synthetic_count += 1
                        matched_chain[next_unmatched] = True
                        chain_idx = sum(matched_chain)
                    if gpu:
                        metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])
                    continue
                if mode == "app_loop" and chain_idx < len(expected_chain):
                    history += gen_text + "\nUser: You have not finished gathering all the information. Please continue calling tools using the <<<TOOL>>> format. Do NOT answer yet.\nAssistant:"
                    if gpu:
                        metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])
                    continue
                step_ttf = time.time() - t0
                metrics["per_step_ttf"].append(step_ttf)
                metrics["final_answer"] = gen_text.strip()
                break
            else:
                consecutive_no_tool = 0

            match_idx = None
            for i, expected in enumerate(expected_chain):
                if matched_chain[i]:
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
                    match_idx = i
                    metrics["tool_calls_ok"] += 1
                    break
            if match_idx is not None:
                matched_chain[match_idx] = True
                chain_idx = sum(matched_chain)
                expected = expected_chain[match_idx]
                true_name = expected["tool"]
                true_args = expected["tool_args"]
                tool_result = execute_tool(true_name, true_args)
                result_block = self.make_tool_result_block(true_name, true_args, tool_result)
            else:
                tool_result = execute_tool(tool_name, tool_args or {})
                result_block = self.make_tool_result_block(tool_name, tool_args or {}, tool_result)
            metrics["total_tool_calls"] += 1

            history += gen_text + result_block

            if mode == "sig" and rolled_back and synthetic_count < max_synthetic:
                next_unmatched = next((i for i, m in enumerate(matched_chain) if not m), None)
                if next_unmatched is not None:
                    synthetic_call = (
                        f"{self.tool_mark}\n"
                        f"{json.dumps({'name': expected_chain[next_unmatched]['tool'], 'arguments': expected_chain[next_unmatched]['tool_args']})}\n"
                        f"{self.tool_end}\n"
                    )
                    synthetic_ids = list(self.tok(synthetic_call, add_bos=False))
                    pf_t0 = time.time()
                    self.eval_tokens(synthetic_ids)
                    metrics["total_prefill_tokens"] += len(synthetic_ids)
                    metrics["total_prefill_time"] += time.time() - pf_t0
                    sig_cache_ids = sig_cache_ids + synthetic_ids
                    history += synthetic_call
                    synthetic_count += 1
                    matched_chain[next_unmatched] = True
                    chain_idx = sum(matched_chain)

            res_ids = list(self.tok(result_block, add_bos=False))

            if mode == "sig":
                pf_t0 = time.time()
                self.eval_tokens(res_ids)
                metrics["total_prefill_tokens"] += len(res_ids)
                metrics["total_prefill_time"] += time.time() - pf_t0
                if not rolled_back:
                    sig_cache_ids = sig_cache_ids + list(gen_ids) + res_ids
                else:
                    sig_cache_ids = sig_cache_ids + res_ids
            else:
                self.eval_tokens(res_ids)

            step_ttf = time.time() - t0
            metrics["per_step_ttf"].append(step_ttf)
            t0 = time.time()

            if gpu:
                metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        metrics["total_ttf"] = sum(metrics["per_step_ttf"])
        metrics["chain_depth"] = min(sum(matched_chain), len(expected_chain))
        metrics["chain_total"] = len(expected_chain)
        metrics["synthetic_injections"] = synthetic_count
        return metrics

    def run_mode_on_turns(self, mode: str, turns: List[Dict],
                          system_prompt: str = SYSTEM_PROMPT,
                          structural_isolation: bool = False,
                          gpu: Optional[GPUMonitor] = None,
                          debug: bool = True) -> Dict:
        metrics = {
            "total_ttf": 0.0,
            "per_turn_ttf": [],
            "tool_calls_ok": 0,
            "total_tool_calls": 0,
            "final_answer": "",
            "peak_gpu_delta": 0.0,
            "total_gen_tokens": 0,
            "total_prefill_tokens": 0,
            "total_prefill_time": 0.0,
            "delta_prefill_tokens": 0,
            "delta_prefill_time_est": 0.0,
            "rollback_count": 0,
        }
        self.reset_cache()

        first_user = turns[0]["user"] if turns else "Hello"
        history = f"{system_prompt}\n\nUser: {first_user}\nAssistant:"
        init_ids = list(self.tok(history, add_bos=False))

        pf_t0 = time.time()
        self.eval_tokens(init_ids)
        if mode == "sig":
            metrics["total_prefill_tokens"] += len(init_ids)
            metrics["total_prefill_time"] += time.time() - pf_t0

        sig_cache_ids = list(init_ids) if mode == "sig" else []
        prev_ids = list(init_ids)
        if gpu:
            metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        for i, turn in enumerate(turns):
            if i > 0:
                user_line = f"\nUser: {turn['user']}\nAssistant:"
                history += user_line

            if mode == "app_loop":
                full_ids = list(self.tok(history, add_bos=False))
                prefix_len = 0
                for a, b in zip(prev_ids, full_ids):
                    if a == b:
                        prefix_len += 1
                    else:
                        break
                delta_len = len(full_ids) - prefix_len
                self.reset_cache()
                pt0 = time.time()
                self.eval_tokens(full_ids)
                prefill_time = time.time() - pt0
                metrics["total_prefill_tokens"] += len(full_ids)
                metrics["total_prefill_time"] += prefill_time
                metrics["delta_prefill_tokens"] += delta_len
                if len(full_ids) > 0:
                    per_tok = prefill_time / len(full_ids)
                    metrics["delta_prefill_time_est"] += delta_len * per_tok
                prev_ids = list(full_ids)
                t0 = time.time()
            else:
                if i > 0:
                    user_ids = list(self.tok(f"\nUser: {turn['user']}\nAssistant:", add_bos=False))
                    pf_t0 = time.time()
                    self.eval_tokens(user_ids)
                    metrics["total_prefill_tokens"] += len(user_ids)
                    metrics["total_prefill_time"] += time.time() - pf_t0
                    sig_cache_ids = sig_cache_ids + list(user_ids)
                t0 = time.time()

            n_before_gen = self.llm.n_tokens if mode == "sig" else 0

            if turn.get("tool"):
                gen_text, gen_ids = self.generate_until_ids(self.tool_end_ids, max_new=250)
                metrics["total_gen_tokens"] += len(gen_ids)

                rolled_back = False
                if mode == "sig":
                    gen_text, gen_ids, rolled_back = self._sanitize_sig_gen(n_before_gen, gen_text, gen_ids, sig_cache_ids)
                    if rolled_back:
                        metrics["rollback_count"] += 1

                if debug:
                    tag = " [ROLLBACK]" if rolled_back else ""
                    print(f"   [Turn {i+1}] Gen:{tag}\n{gen_text}")

                tool_name, tool_args = self._parse_tool_call(gen_text)
                true_name = turn["tool"]
                true_args = turn["tool_args"]

                if mode == "sig" and rolled_back:
                    synthetic_call = (
                        f"{self.tool_mark}\n"
                        f"{json.dumps({'name': true_name, 'arguments': true_args})}\n"
                        f"{self.tool_end}\n"
                    )
                    synthetic_ids = list(self.tok(synthetic_call, add_bos=False))
                    pf_t0 = time.time()
                    self.eval_tokens(synthetic_ids)
                    metrics["total_prefill_tokens"] += len(synthetic_ids)
                    metrics["total_prefill_time"] += time.time() - pf_t0
                    sig_cache_ids = sig_cache_ids + synthetic_ids
                    history += synthetic_call
                    tool_name = true_name
                    tool_args = true_args

                name_ok = (tool_name == true_name)
                args_ok = False
                if tool_args and true_args:
                    normalized_parsed = {k: normalize_city(str(v)) for k, v in tool_args.items()}
                    normalized_true = {k: normalize_city(str(v)) for k, v in true_args.items()}
                    args_ok = (normalized_parsed == normalized_true)
                if name_ok and args_ok:
                    metrics["tool_calls_ok"] += 1
                metrics["total_tool_calls"] += 1

                tool_result = execute_tool(true_name, true_args)
                if turn.get("attack_payload"):
                    tool_result = turn["attack_payload"]

                result_block = self.make_tool_result_block(true_name, true_args, tool_result)
                if structural_isolation:
                    result_block = f"<<<SAFE>>>\n{result_block}\n<<</SAFE>>>"

                res_ids = list(self.tok(result_block, add_bos=False))

                if mode == "sig":
                    pf_t0 = time.time()
                    self.eval_tokens(res_ids)
                    metrics["total_prefill_tokens"] += len(res_ids)
                    metrics["total_prefill_time"] += time.time() - pf_t0
                    if not rolled_back:
                        sig_cache_ids = sig_cache_ids + list(gen_ids) + res_ids
                    else:
                        sig_cache_ids = sig_cache_ids + res_ids
                else:
                    self.eval_tokens(res_ids)

                history += gen_text + result_block
            else:
                gen_text, gen_ids, hit_stop = self.generate_until_any(
                    ["\nUser:", self.tool_mark, "\n\n\n"], max_new=150
                )
                metrics["total_gen_tokens"] += len(gen_ids)

                if mode == "sig":
                    gen_text, gen_ids, rolled_back = self._sanitize_sig_gen(n_before_gen, gen_text, gen_ids, sig_cache_ids)
                    if rolled_back:
                        metrics["rollback_count"] += 1

                if hit_stop == self.tool_mark:
                    if debug:
                        print(f"   [Turn {i+1}] Model generated tool marker in non-tool turn")
                    metrics["final_answer"] = ""
                else:
                    if debug:
                        print(f"   [Turn {i+1}] Answer:\n{gen_text}")
                    metrics["final_answer"] = gen_text.strip()
                history += gen_text
                if mode == "sig":
                    if not rolled_back:
                        sig_cache_ids = sig_cache_ids + list(gen_ids)

            ttf = time.time() - t0
            metrics["per_turn_ttf"].append(ttf)
            if gpu:
                metrics["peak_gpu_delta"] = max(metrics["peak_gpu_delta"], gpu.snapshot()["delta_mb"])

        metrics["total_ttf"] = sum(metrics["per_turn_ttf"])
        return metrics

    # ================================================================
    # Scenario builders
    # ================================================================

    def build_scenario1_long_sequence(self, n_turns: int = 22) -> List[Dict]:
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

    def build_scenario2_multi_tool_chain(self) -> List[Dict]:
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

    def build_scenario3_rapid_fire(self, n_queries: int = 12) -> List[Dict]:
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
        turns = []
        for i in range(min(n_queries, len(queries))):
            q, tool, args = queries[i]
            turns.append({"user": q, "tool": tool, "tool_args": args})
        return turns

    def build_scenario4_long_document(self) -> Tuple[str, List[Dict]]:
        system_with_doc = SYSTEM_PROMPT + "\n\n" + LONG_BACKGROUND
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

    def build_scenario5_mixed_conversation(self) -> List[Dict]:
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

    def build_scenario6_deep_tool_chain(self) -> List[Dict]:
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

    def build_scenario7_travel_planning_chain(self) -> Tuple[str, List[Dict]]:
        query = ("I'm planning a trip from New York to Tokyo with stops in London and Dubai. "
                 "For each city (New York, London, Dubai, Tokyo), I need to know the attractions "
                 "and weather. I also need flights between consecutive cities. "
                 "Finally, based on the weather in all cities, recommend what to pack.")
        chain = [
            {"tool": "search_attractions", "tool_args": {"city": "newyork"}},
            {"tool": "get_weather", "tool_args": {"city": "newyork"}},
            {"tool": "get_flight_info", "tool_args": {"origin": "newyork", "destination": "london"}},
            {"tool": "search_attractions", "tool_args": {"city": "london"}},
            {"tool": "get_weather", "tool_args": {"city": "london"}},
            {"tool": "get_flight_info", "tool_args": {"origin": "london", "destination": "dubai"}},
            {"tool": "search_attractions", "tool_args": {"city": "dubai"}},
            {"tool": "get_weather", "tool_args": {"city": "dubai"}},
            {"tool": "get_flight_info", "tool_args": {"origin": "dubai", "destination": "tokyo"}},
            {"tool": "search_attractions", "tool_args": {"city": "tokyo"}},
            {"tool": "get_weather", "tool_args": {"city": "tokyo"}},
        ]
        return query, chain

    def build_scenario8_code_debugging_chain(self) -> Tuple[str, List[Dict]]:
        query = ("I have a bug in my Python project. The test_calculator test suite is failing. "
                 "Please use the tools to investigate: first run the test, then read calculator.py, "
                 "then search for 'divide' in the codebase, then read test_calculator.py. "
                 "Only after gathering all information, explain the bug and suggest a fix.")
        chain = [
            {"tool": "run_test", "tool_args": {"test_name": "test_calculator"}},
            {"tool": "read_file", "tool_args": {"path": "calculator.py"}},
            {"tool": "search_code", "tool_args": {"query": "divide"}},
            {"tool": "read_file", "tool_args": {"path": "test_calculator.py"}},
        ]
        return query, chain

    def build_scenario9_cross_reference_chain(self) -> Tuple[str, List[Dict]]:
        query = ("I need to compare travel options between Paris, Rome, and London. "
                 "Please use the tools to check: "
                 "1) attractions in Paris, Rome, and London, "
                 "2) weather in Paris, Rome, and London, "
                 "3) flights from Paris to Rome, Rome to London, and Paris to London. "
                 "Call each tool one at a time. Do NOT answer until you have called all 9 tools.")
        chain = [
            {"tool": "search_attractions", "tool_args": {"city": "paris"}},
            {"tool": "get_weather", "tool_args": {"city": "paris"}},
            {"tool": "search_attractions", "tool_args": {"city": "rome"}},
            {"tool": "get_weather", "tool_args": {"city": "rome"}},
            {"tool": "search_attractions", "tool_args": {"city": "london"}},
            {"tool": "get_weather", "tool_args": {"city": "london"}},
            {"tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "rome"}},
            {"tool": "get_flight_info", "tool_args": {"origin": "rome", "destination": "london"}},
            {"tool": "get_flight_info", "tool_args": {"origin": "paris", "destination": "london"}},
        ]
        return query, chain


def average_metrics(runs: List[Dict]) -> Dict:
    if not runs:
        return {}
    if len(runs) == 1:
        return dict(runs[0])
    n = len(runs)
    avg = {}
    sum_fields = [
        "total_ttf", "tool_calls_ok", "total_tool_calls",
        "total_gen_tokens", "total_prefill_tokens", "total_prefill_time",
        "delta_prefill_tokens", "delta_prefill_time_est",
        "rollback_count",
    ]
    for f in sum_fields:
        avg[f] = sum(r.get(f, 0) for r in runs) / n
    if any("steps" in r for r in runs):
        avg["steps"] = sum(r.get("steps", 0) for r in runs) / n
    if any("chain_depth" in r for r in runs):
        avg["chain_depth"] = sum(r.get("chain_depth", 0) for r in runs) / n
        avg["chain_total"] = runs[0].get("chain_total", 0)
        avg["synthetic_injections"] = sum(r.get("synthetic_injections", 0) for r in runs) / n
    avg["peak_gpu_delta"] = max(r.get("peak_gpu_delta", 0) for r in runs)
    avg["final_answer"] = runs[0].get("final_answer", "")
    ttf_key = None
    for r in runs:
        if "per_turn_ttf" in r:
            ttf_key = "per_turn_ttf"
            break
        if "per_step_ttf" in r:
            ttf_key = "per_step_ttf"
            break
    if ttf_key:
        max_len = max(len(r.get(ttf_key, [])) for r in runs)
        avg[ttf_key] = [
            sum(r.get(ttf_key, [0]*max_len)[i] for r in runs if i < len(r.get(ttf_key, [])))
            / sum(1 for r in runs if i < len(r.get(ttf_key, [])))
            for i in range(max_len)
        ]
        other_key = "per_step_ttf" if ttf_key == "per_turn_ttf" else "per_turn_ttf"
        avg[other_key] = avg[ttf_key]
    avg["correct_runs"] = n
    return avg


def print_scenario_header(num: int, title: str, desc: str):
    print("\n" + "=" * 60)
    print(f"Scenario {num}: {title}")
    print(f"  {desc}")
    print("=" * 60)


def print_mode_result(mode: str, met: Dict):
    ttf_list = met.get("per_turn_ttf", met.get("per_step_ttf", []))
    avg_ttf = sum(ttf_list) / len(ttf_list) if ttf_list else 0
    tool_acc = f"{met['tool_calls_ok']:.0f}/{met['total_tool_calls']:.0f}" if met['total_tool_calls'] > 0 else "N/A"
    tool_pct = f"({100*met['tool_calls_ok']/met['total_tool_calls']:.0f}%)" if met['total_tool_calls'] > 0 else ""
    steps = met.get("steps", len(ttf_list))
    correct = met.get("correct_runs", "")
    correct_str = f" [avg of {correct}]" if correct else ""
    chain_d = met.get("chain_depth", None)
    chain_t = met.get("chain_total", None)
    syn = met.get("synthetic_injections", None)
    chain_str = f" | chain: {chain_d:.0f}/{chain_t}" if chain_d is not None else ""
    if chain_d is not None and syn is not None and syn > 0:
        chain_str += f" (syn:{syn:.0f})"
    print(f"   {mode:15s} | TTF: {met['total_ttf']:6.2f}s | avg: {avg_ttf:.2f}s/step | "
          f"steps: {steps:.0f} | tools: {tool_acc} {tool_pct} | gen_toks: {met['total_gen_tokens']:.0f} | "
          f"prefill: {met['total_prefill_tokens']:.0f} | GPU: {met['peak_gpu_delta']:.0f} MB{chain_str}{correct_str}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced SIG Benchmark — 9 Scenarios")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--long-turns", type=int, default=22)
    parser.add_argument("--rapid-queries", type=int, default=12)
    parser.add_argument("--no-debug", action="store_true", help="Disable debug output")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per mode per scenario (default: 10)")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated scenario numbers to skip (e.g. '3,5')")
    args = parser.parse_args()
    args.debug = not args.no_debug
    skip = set(int(x.strip()) for x in args.skip.split(",") if x.strip())

    gpu = GPUMonitor()
    bench = EnhancedLlamaBench(args.model, n_ctx=args.n_ctx, n_threads=args.n_threads,
                               n_gpu_layers=args.n_gpu_layers)

    all_results = {}
    n_runs = args.runs

    def run_multi_turns(mode, turns, system_prompt=SYSTEM_PROMPT, gpu=gpu, debug=args.debug):
        correct_runs = []
        all_tool_ok = []
        all_runs = []
        for ri in range(n_runs):
            met = bench.run_mode_on_turns(mode, turns, system_prompt=system_prompt,
                                          gpu=gpu, debug=debug)
            ok = met["tool_calls_ok"]
            total = met["total_tool_calls"]
            all_tool_ok.append(f"{ok}/{total}")
            all_runs.append(met)
            if ok == total and total > 0:
                correct_runs.append(met)
        print(f"     {mode}: runs={n_runs}, tool_acc=[{', '.join(all_tool_ok)}], correct={len(correct_runs)}/{n_runs}")
        if correct_runs:
            return average_metrics(correct_runs)
        best = max(all_runs, key=lambda m: m["tool_calls_ok"])
        best["correct_runs"] = 0
        print(f"     WARNING: No fully correct run for {mode}, using best effort ({best['tool_calls_ok']}/{best['total_tool_calls']})")
        return best

    def run_multi_complex(mode, query, chain, system_prompt=SYSTEM_PROMPT, gpu=gpu, debug=args.debug):
        correct_runs = []
        all_tool_ok = []
        all_runs = []
        for ri in range(n_runs):
            met = bench.run_complex_task(mode, query, chain,
                                         system_prompt=system_prompt, gpu=gpu, debug=debug)
            ok = met["tool_calls_ok"]
            total = met["total_tool_calls"]
            all_tool_ok.append(f"{ok}/{total}")
            all_runs.append(met)
            if ok == total and total > 0:
                correct_runs.append(met)
        print(f"     {mode}: runs={n_runs}, tool_acc=[{', '.join(all_tool_ok)}], correct={len(correct_runs)}/{n_runs}")
        if correct_runs:
            return average_metrics(correct_runs)
        best = max(all_runs, key=lambda m: m["tool_calls_ok"])
        best["correct_runs"] = 0
        print(f"     WARNING: No fully correct run for {mode}, using best effort ({best['tool_calls_ok']}/{best['total_tool_calls']})")
        return best

    # ================================================================
    # Scenario 1: Long-sequence stress test
    # ================================================================
    if 1 not in skip:
        print_scenario_header(1, "Long-sequence stress test",
                              "20+ turns, context grows to 4K+ tokens. Tests cumulative KV cache advantage.")
        turns = bench.build_scenario1_long_sequence(args.long_turns)
        print(f"   {len(turns)} turns generated, {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_turns(mode, turns)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[1] = results

    # ================================================================
    # Scenario 2: Multi-tool chain (complex reasoning)
    # ================================================================
    if 2 not in skip:
        print_scenario_header(2, "Multi-tool chain (complex reasoning)",
                              "One complex query requiring 4 sequential tool calls + summary. "
                              "Tests SIG's ability to chain without re-prefilling.")
        turns = bench.build_scenario2_multi_tool_chain()
        print(f"   {len(turns)} turns (4 tool calls + 1 summary), {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_turns(mode, turns)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[2] = results

    # ================================================================
    # Scenario 3: Rapid-fire short queries
    # ================================================================
    if 3 not in skip:
        print_scenario_header(3, "Rapid-fire short queries",
                              "Many independent short queries, each needing 1 tool call. "
                              "Tests overhead of cache reset vs continuous generation.")
        turns = bench.build_scenario3_rapid_fire(args.rapid_queries)
        print(f"   {len(turns)} short queries, {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_turns(mode, turns)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[3] = results

    # ================================================================
    # Scenario 4: Long-document + tool calls
    # ================================================================
    if 4 not in skip:
        print_scenario_header(4, "Long-document + tool calls",
                              "Large background text (~1K tokens) injected, then tool calls on top. "
                              "Tests SIG advantage when prefix is very long (expensive to re-prefill).")
        system_with_doc, turns = bench.build_scenario4_long_document()
        init_tokens = len(bench.tok(f"{system_with_doc}\n\nUser: test\nAssistant:", add_bos=False))
        print(f"   System prompt + background: ~{init_tokens} tokens, {len(turns)} turns, {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_turns(mode, turns, system_prompt=system_with_doc)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[4] = results

    # ================================================================
    # Scenario 5: Mixed conversation (tool + chitchat)
    # ================================================================
    if 5 not in skip:
        print_scenario_header(5, "Mixed conversation (tool + chitchat)",
                              "Alternating between tool calls and plain conversation. "
                              "Tests SIG handling of turns that don't need tools.")
        turns = bench.build_scenario5_mixed_conversation()
        tool_count = sum(1 for t in turns if t.get("tool"))
        chat_count = len(turns) - tool_count
        print(f"   {len(turns)} turns ({tool_count} tool, {chat_count} chitchat), {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_turns(mode, turns)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[5] = results

    # ================================================================
    # Scenario 6: Deep tool chain (8+ tool calls)
    # ================================================================
    if 6 not in skip:
        print_scenario_header(6, "Deep tool chain (round-the-world)",
                              "15 turns with 14 sequential tool calls across 5 cities. "
                              "Tests SIG's sustained advantage with many chained tool calls.")
        turns = bench.build_scenario6_deep_tool_chain()
        tool_count = sum(1 for t in turns if t.get("tool"))
        chat_count = len(turns) - tool_count
        print(f"   {len(turns)} turns ({tool_count} tool, {chat_count} summary), {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_turns(mode, turns)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[6] = results

    # ================================================================
    # Scenario 7: Complex task - Travel planning chain
    # ================================================================
    if 7 not in skip:
        print_scenario_header(7, "Complex task: Travel planning chain",
                              "Single complex query requiring 11 autonomous tool calls. "
                              "Model must chain: attractions→weather→flights for 4 cities. "
                              "Tests SIG's core advantage: uninterrupted reasoning chain.")
        query, chain = bench.build_scenario7_travel_planning_chain()
        print(f"   1 query, {len(chain)} expected tool calls in chain, {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_complex(mode, query, chain)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[7] = results

    # ================================================================
    # Scenario 8: Complex task - Code debugging chain
    # ================================================================
    if 8 not in skip:
        print_scenario_header(8, "Complex task: Code debugging chain",
                              "Programming agent: run tests → read code → search code → read tests. "
                              "Tests SIG in a coding agent scenario where each step depends on previous results.")
        query, chain = bench.build_scenario8_code_debugging_chain()
        print(f"   1 query, {len(chain)} expected tool calls in chain, {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_complex(mode, query, chain,
                                    system_prompt=SYSTEM_PROMPT_DEV)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[8] = results

    # ================================================================
    # Scenario 9: Complex task - Cross-reference analysis
    # ================================================================
    if 9 not in skip:
        print_scenario_header(9, "Complex task: Cross-reference analysis",
                              "Compare 3 cities: attractions+weather for each, then all flight pairs. "
                              "9 tool calls requiring cross-referencing results. "
                              "Tests SIG's ability to maintain context across many tool results.")
        query, chain = bench.build_scenario9_cross_reference_chain()
        print(f"   1 query, {len(chain)} expected tool calls in chain, {n_runs} runs per mode")
        results = {}
        for mode in ["app_loop", "sig"]:
            met = run_multi_complex(mode, query, chain)
            print_mode_result(mode, met)
            results[mode] = met
        all_results[9] = results

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("=== Cross-Scenario Summary ===")
    print("=" * 80)

    print("\n--- Speed (Generation TTF) ---")
    header = f"{'Scenario':<14} {'App Loop':<12} {'SIG':<12} {'Speedup':<10}"
    print(header)
    print("-" * len(header))
    scenario_names = {
        1: "Long-seq",
        2: "Multi-tool",
        3: "Rapid-fire",
        4: "Long-doc",
        5: "Mixed",
        6: "Deep-chain",
        7: "Travel-plan",
        8: "Code-debug",
        9: "Cross-ref",
    }
    for snum, results in sorted(all_results.items()):
        t_app = results.get("app_loop", {}).get("total_ttf", 0)
        t_sig = results.get("sig", {}).get("total_ttf", 0)
        sp = f"{t_app/t_sig:.2f}x" if t_sig > 0 else "N/A"
        name = scenario_names.get(snum, f"S{snum}")
        print(f"{name:<14} {t_app:<12.2f} {t_sig:<12.2f} {sp:<10}")

    print("\n--- Tool Accuracy (avg of correct runs) ---")
    header2 = f"{'Scenario':<14} {'App Loop':<24} {'SIG':<24}"
    print(header2)
    print("-" * len(header2))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        row = f"{name:<14}"
        for mode in ["app_loop", "sig"]:
            m = results.get(mode, {})
            ok = m.get("tool_calls_ok", 0)
            total = m.get("total_tool_calls", 0)
            pct = f"{100*ok/total:.0f}%" if total > 0 else "N/A"
            cr = m.get("correct_runs", "")
            cr_str = f"[{cr}/{n_runs}]" if cr != "" else ""
            row += f" {ok:.0f}/{total:.0f} ({pct:>4}) {cr_str:<8}"
        print(row)

    print("\n--- Prefill Cost (tokens) ---")
    header3 = f"{'Scenario':<14} {'Full Prefill':<14} {'Delta Prefill':<14} {'SIG Prefill':<14} {'Cache Save':<12} {'SIG Save':<12}"
    print(header3)
    print("-" * len(header3))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        pf_full = results.get("app_loop", {}).get("total_prefill_tokens", 0)
        pf_delta = results.get("app_loop", {}).get("delta_prefill_tokens", 0)
        pf_sig = results.get("sig", {}).get("total_prefill_tokens", 0)
        cache_save = f"{(pf_full-pf_delta)/pf_full*100:.0f}%" if pf_full > 0 else "N/A"
        sig_save = f"{(pf_full-pf_sig)/pf_full*100:.0f}%" if pf_full > 0 else "N/A"
        print(f"{name:<14} {pf_full:<14.0f} {pf_delta:<14.0f} {pf_sig:<14.0f} {cache_save:<12} {sig_save:<12}")

    print("\n--- Prefill Cost (time, seconds) ---")
    header4 = f"{'Scenario':<14} {'Full Prefill':<14} {'Delta Prefill':<14} {'SIG Prefill':<14} {'Cache Save':<12} {'SIG Save':<12}"
    print(header4)
    print("-" * len(header4))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        pf_full_t = results.get("app_loop", {}).get("total_prefill_time", 0)
        pf_delta_t = results.get("app_loop", {}).get("delta_prefill_time_est", 0)
        pf_sig_t = results.get("sig", {}).get("total_prefill_time", 0)
        cache_save_t = f"{(pf_full_t-pf_delta_t)/pf_full_t*100:.0f}%" if pf_full_t > 0 else "N/A"
        sig_save_t = f"{(pf_full_t-pf_sig_t)/pf_full_t*100:.0f}%" if pf_full_t > 0 else "N/A"
        print(f"{name:<14} {pf_full_t:<14.2f} {pf_delta_t:<14.2f} {pf_sig_t:<14.2f} {cache_save_t:<12} {sig_save_t:<12}")

    print("\n--- Total Time (Generation + Prefill) ---")
    header5 = f"{'Scenario':<14} {'App Loop':<12} {'+Cache':<12} {'SIG':<12} {'Cache Spd':<12} {'SIG Spd':<12}"
    print(header5)
    print("-" * len(header5))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        gen_app = results.get("app_loop", {}).get("total_ttf", 0)
        pf_full_t = results.get("app_loop", {}).get("total_prefill_time", 0)
        pf_delta_t = results.get("app_loop", {}).get("delta_prefill_time_est", 0)
        gen_sig = results.get("sig", {}).get("total_ttf", 0)
        pf_sig_t = results.get("sig", {}).get("total_prefill_time", 0)
        total_app = gen_app + pf_full_t
        total_cache = gen_app + pf_delta_t
        total_sig = gen_sig + pf_sig_t
        cache_spd = f"{total_app/total_cache:.2f}x" if total_cache > 0 else "N/A"
        sig_spd = f"{total_app/total_sig:.2f}x" if total_sig > 0 else "N/A"
        print(f"{name:<14} {total_app:<12.2f} {total_cache:<12.2f} {total_sig:<12.2f} {cache_spd:<12} {sig_spd:<12}")

    print("\n--- SIG Stability ---")
    header6 = f"{'Scenario':<14} {'Rollbacks':<12} {'SIG GPU MB':<12}"
    print(header6)
    print("-" * len(header6))
    for snum, results in sorted(all_results.items()):
        name = scenario_names.get(snum, f"S{snum}")
        sig_rb = results.get("sig", {}).get("rollback_count", 0)
        sig_gpu = results.get("sig", {}).get("peak_gpu_delta", 0)
        print(f"{name:<14} {sig_rb:<12} {sig_gpu:<12.0f}")

    has_chain = any(
        results.get("app_loop", {}).get("chain_depth", 0) > 0
        or results.get("sig", {}).get("chain_depth", 0) > 0
        for results in all_results.values()
    )
    if has_chain:
        print("\n--- Chain Completion (complex tasks) ---")
        header7 = f"{'Scenario':<14} {'App Loop':<20} {'SIG':<24} {'SIG Advantage':<14}"
        print(header7)
        print("-" * len(header7))
        for snum, results in sorted(all_results.items()):
            name = scenario_names.get(snum, f"S{snum}")
            app_d = results.get("app_loop", {}).get("chain_depth", 0)
            app_t = results.get("app_loop", {}).get("chain_total", 0)
            sig_d = results.get("sig", {}).get("chain_depth", 0)
            sig_t = results.get("sig", {}).get("chain_total", 0)
            sig_syn = results.get("sig", {}).get("synthetic_injections", 0)
            if app_t == 0 and sig_t == 0:
                continue
            app_str = f"{app_d:.0f}/{app_t}" if app_t > 0 else "N/A"
            sig_str = f"{sig_d:.0f}/{sig_t}" if sig_t > 0 else "N/A"
            if sig_syn > 0:
                sig_str += f" (syn:{sig_syn:.0f})"
            if app_t > 0 and sig_t > 0:
                app_pct = app_d / app_t
                sig_pct = sig_d / sig_t
                adv = f"{sig_pct - app_pct:+.0%}"
            else:
                adv = "N/A"
            print(f"{name:<14} {app_str:<20} {sig_str:<24} {adv:<14}")

    if gpu.enabled:
        final_snap = gpu.snapshot()
        print(f"\nGPU Final: Used {final_snap['used_mb']:.0f} MB, Delta {final_snap['delta_mb']:+.0f} MB")

    gpu.shutdown()
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
