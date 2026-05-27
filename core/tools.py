"""ToolRegistry — simulated tool execution for benchmark scenarios.

Merged from co_benchmark.py (dev tools: read_file, search_code, run_test)
and r2_benchmark.py (extended travel tools with more cities and routes).

The ``execute`` method uses ``normalize_city`` for city-name lookups so
that aliases like "New York" or "NY" resolve correctly.
"""

from .text_utils import normalize_city


class ToolRegistry:
    """Registry of simulated tools with static return values.

    Travel tools:
        search_attractions, get_weather, get_flight_info

    Dev tools:
        read_file, search_code, run_test
    """

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
                "seoul": "1. Gyeongbokgung. 2. N Seoul Tower. 3. Myeongdong. 4. Bukchon Hanok Village. 5. Dongdaemun.",
                "bangkok": "1. Grand Palace. 2. Wat Pho. 3. Khao San Road. 4. Chatuchak Market. 5. Wat Arun.",
                "istanbul": "1. Hagia Sophia. 2. Blue Mosque. 3. Topkapi Palace. 4. Grand Bazaar. 5. Bosphorus Cruise.",
                "cairo": "1. Pyramids of Giza. 2. Egyptian Museum. 3. Khan el-Khalili. 4. Al-Azhar Mosque. 5. Citadel.",
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
                "seoul": "Clear, 20C",
                "bangkok": "Humid, 33C",
                "istanbul": "Mild, 21C",
                "cairo": "Hot, 35C",
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
                "paris_seoul": "Direct: KE918 (10h30m, $780)",
                "seoul_tokyo": "Direct: JL952 (2h30m, $250)",
                "bangkok_dubai": "Direct: EK373 (6h30m, $550)",
                "istanbul_cairo": "Direct: TK690 (2h, $200)",
                "cairo_dubai": "Direct: EK926 (3h30m, $400)",
            },
            "read_file": {
                "calculator.py": (
                    "class Calculator:\n"
                    "    def __init__(self):\n"
                    "        self.history = []\n"
                    "    def add(self, a, b):\n"
                    "        result = a + b\n"
                    "        self.history.append(f'{a}+{b}={result}')\n"
                    "        return result\n"
                    "    def subtract(self, a, b):\n"
                    "        result = a - b\n"
                    "        self.history.append(f'{a}-{b}={result}')\n"
                    "        return result\n"
                    "    def multiply(self, a, b):\n"
                    "        result = a * b\n"
                    "        self.history.append(f'{a}*{b}={result}')\n"
                    "        return result\n"
                    "    def divide(self, a, b):\n"
                    "        result = a / b\n"
                    "        self.history.append(f'{a}/{b}={result}')\n"
                    "        return result\n"
                    "    def power(self, a, b):\n"
                    "        result = a ** b\n"
                    "        self.history.append(f'{a}**{b}={result}')\n"
                    "        return result"
                ),
                "test_calculator.py": (
                    "import unittest\n"
                    "from calculator import Calculator\n"
                    "\n"
                    "class TestCalculator(unittest.TestCase):\n"
                    "    def setUp(self):\n"
                    "        self.calc = Calculator()\n"
                    "    def test_add(self):\n"
                    "        self.assertEqual(self.calc.add(2, 3), 5)\n"
                    "    def test_subtract(self):\n"
                    "        self.assertEqual(self.calc.subtract(5, 3), 2)\n"
                    "    def test_multiply(self):\n"
                    "        self.assertEqual(self.calc.multiply(4, 3), 12)\n"
                    "    def test_divide(self):\n"
                    "        self.assertEqual(self.calc.divide(10, 2), 5)\n"
                    "    def test_divide_by_zero(self):\n"
                    "        with self.assertRaises(ZeroDivisionError):\n"
                    "            self.calc.divide(10, 0)\n"
                    "    def test_power(self):\n"
                    "        self.assertEqual(self.calc.power(2, 3), 8)\n"
                    "    def test_history(self):\n"
                    "        self.calc.add(1, 2)\n"
                    "        self.calc.multiply(3, 4)\n"
                    "        self.assertEqual(len(self.calc.history), 2)"
                ),
                "auth.py": (
                    "import hashlib\n"
                    "import secrets\n"
                    "\n"
                    "class AuthManager:\n"
                    "    def __init__(self):\n"
                    "        self.users = {}\n"
                    "    def register(self, username, password):\n"
                    "        salt = secrets.token_hex(16)\n"
                    "        hashed = hashlib.sha256((password + salt).encode()).hexdigest()\n"
                    "        self.users[username] = {'salt': salt, 'hash': hashed}\n"
                    "        return True\n"
                    "    def login(self, username, password):\n"
                    "        if username not in self.users:\n"
                    "            return False\n"
                    "        user = self.users[username]\n"
                    "        hashed = hashlib.sha256((password + user['salt']).encode()).hexdigest()\n"
                    "        return hashed == user['hash']\n"
                    "    def change_password(self, username, old_pw, new_pw):\n"
                    "        if not self.login(username, old_pw):\n"
                    "            return False\n"
                    "        salt = secrets.token_hex(16)\n"
                    "        hashed = hashlib.sha256((new_pw + salt).encode()).hexdigest()\n"
                    "        self.users[username] = {'salt': salt, 'hash': hashed}\n"
                    "        return True"
                ),
                "config.py": (
                    "import json\n"
                    "import os\n"
                    "\n"
                    "CONFIG_PATH = '/etc/app/config.json'\n"
                    "DEFAULT_CONFIG = {\n"
                    "    'debug': False,\n"
                    "    'port': 8080,\n"
                    "    'database': {'host': 'localhost', 'port': 5432, 'name': 'app_db'},\n"
                    "    'cache': {'enabled': True, 'ttl': 3600},\n"
                    "    'logging': {'level': 'INFO', 'file': '/var/log/app.log'}\n"
                    "}\n"
                    "\n"
                    "def load_config():\n"
                    "    if os.path.exists(CONFIG_PATH):\n"
                    "        with open(CONFIG_PATH) as f:\n"
                    "            return json.load(f)\n"
                    "    return DEFAULT_CONFIG.copy()\n"
                    "\n"
                    "def get_db_url(config=None):\n"
                    "    cfg = config or load_config()\n"
                    "    db = cfg['database']\n"
                    "    return f\"postgresql://{db['host']}:{db['port']}/{db['name']}\""
                ),
                "api.py": (
                    "from flask import Flask, jsonify, request\n"
                    "from auth import AuthManager\n"
                    "from config import load_config\n"
                    "\n"
                    "app = Flask(__name__)\n"
                    "auth = AuthManager()\n"
                    "config = load_config()\n"
                    "\n"
                    "@app.route('/api/register', methods=['POST'])\n"
                    "def register():\n"
                    "    data = request.json\n"
                    "    return jsonify({'success': auth.register(data['username'], data['password'])})\n"
                    "\n"
                    "@app.route('/api/login', methods=['POST'])\n"
                    "def login():\n"
                    "    data = request.json\n"
                    "    return jsonify({'success': auth.login(data['username'], data['password'])})\n"
                    "\n"
                    "@app.route('/api/config', methods=['GET'])\n"
                    "def get_config():\n"
                    "    return jsonify(config)\n"
                    "\n"
                    "@app.route('/api/health', methods=['GET'])\n"
                    "def health():\n"
                    "    return jsonify({'status': 'ok', 'debug': config.get('debug', False)})"
                ),
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
                "test_calculator": (
                    "FAILED: test_divide_by_zero - Expected ZeroDivisionError but got 5.0 "
                    "(divide(10,0) returned inf instead of raising)\n"
                    "PASSED: test_add - 2+3=5\n"
                    "PASSED: test_subtract - 5-3=2\n"
                    "PASSED: test_multiply - 4*3=12\n"
                    "PASSED: test_divide - 10/2=5\n"
                    "PASSED: test_power - 2**3=8\n"
                    "PASSED: test_history\n"
                    "1 FAILED, 6 PASSED out of 7 tests"
                ),
                "test_auth": (
                    "PASSED: test_register\n"
                    "PASSED: test_login_success\n"
                    "PASSED: test_login_wrong_password\n"
                    "PASSED: test_change_password\n"
                    "4 PASSED out of 4 tests"
                ),
                "test_api": (
                    "FAILED: test_health_endpoint - Expected status=ok, got 500 Internal Server Error\n"
                    "FAILED: test_config_endpoint - Missing 'cache' key in response\n"
                    "PASSED: test_register_endpoint\n"
                    "PASSED: test_login_endpoint\n"
                    "2 FAILED, 2 PASSED out of 4 tests"
                ),
                "test_all": (
                    "TOTAL: 3 FAILED, 12 PASSED out of 15 tests\n"
                    "Failures: test_divide_by_zero, test_health_endpoint, test_config_endpoint"
                ),
            },
        }

    def execute(self, name: str, args: dict) -> str:
        """Execute a simulated tool call and return the static result string.

        City-name arguments are normalized via ``normalize_city`` so that
        aliases like "New York" or "NY" resolve to the canonical key.
        """
        if name == "search_attractions":
            key = normalize_city(args.get("city", ""))
            return self.tools["search_attractions"].get(key, f"No data for {key}")
        elif name == "get_weather":
            key = normalize_city(args.get("city", ""))
            return self.tools["get_weather"].get(key, f"No data for {key}")
        elif name == "get_flight_info":
            o = normalize_city(args.get("origin", ""))
            d = normalize_city(args.get("destination", ""))
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


class LatencyToolWrapper:
    """Wraps any tool executor with a configurable per-call delay.

    Simulates realistic network/processing latency for tool execution.
    Used for latency ablation experiments to determine how SIG advantage
    changes under real-world conditions.

    NOTE: Uses ``time.sleep()`` which blocks the calling thread and does
    not release the GIL — this approximates synchronous I/O latency but
    does not simulate the GIL-release behaviour of real network I/O or
    the jitter characteristic of cloud API calls.

    Usage::

        tools = ToolRegistry()
        delayed_tools = LatencyToolWrapper(tools, delay_ms=300)
        result = delayed_tools.execute("get_weather", {"city": "paris"})
    """

    def __init__(self, inner, delay_ms: float = 0.0):
        self._inner = inner
        self.delay_ms = delay_ms
        self._call_count = 0
        self._total_delay_s = 0.0

    def execute(self, name: str, args: dict) -> str:
        if self.delay_ms > 0:
            import time
            time.sleep(self.delay_ms / 1000.0)
            self._call_count += 1
            self._total_delay_s += self.delay_ms / 1000.0
        return self._inner.execute(name, args)

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_delay_s(self) -> float:
        return self._total_delay_s
