import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import ToolRegistry
from core.tools import LatencyToolWrapper


class TestToolRegistry(unittest.TestCase):

    def test_search_attractions(self):
        tr = ToolRegistry()
        result = tr.execute("search_attractions", {"city": "paris"})
        self.assertIn("Eiffel Tower", result)
        self.assertIn("Louvre", result)

    def test_get_weather(self):
        tr = ToolRegistry()
        result = tr.execute("get_weather", {"city": "rome"})
        self.assertTrue("26C" in result or "Sunny" in result)

    def test_get_flight_info(self):
        tr = ToolRegistry()
        result = tr.execute("get_flight_info", {"origin": "paris", "destination": "rome"})
        self.assertIn("AF1104", result)

    def test_read_file(self):
        tr = ToolRegistry()
        result = tr.execute("read_file", {"path": "calculator.py"})
        self.assertIn("class Calculator", result)

    def test_search_code(self):
        tr = ToolRegistry()
        result = tr.execute("search_code", {"query": "divide"})
        self.assertIn("calculator.py", result)

    def test_run_test(self):
        tr = ToolRegistry()
        result = tr.execute("run_test", {"test_name": "test_calculator"})
        self.assertIn("FAILED", result)
        self.assertIn("PASSED", result)

    def test_unknown_tool(self):
        tr = ToolRegistry()
        result = tr.execute("nonexistent_tool", {})
        self.assertIn("Unknown tool", result)

    def test_missing_key(self):
        tr = ToolRegistry()
        result = tr.execute("get_weather", {"city": "atlantis"})
        self.assertIn("No data for atlantis", result)

    def test_city_normalization(self):
        tr = ToolRegistry()
        r1 = tr.execute("get_weather", {"city": "New York"})
        r2 = tr.execute("get_weather", {"city": "newyork"})
        self.assertEqual(r1, r2)


class TestLatencyToolWrapper(unittest.TestCase):

    def test_zero_delay(self):
        tr = ToolRegistry()
        wrapper = LatencyToolWrapper(tr, delay_ms=0)
        result = wrapper.execute("get_weather", {"city": "paris"})
        self.assertIn("18C", result)
        self.assertEqual(wrapper.call_count, 0)

    def test_with_delay(self):
        tr = ToolRegistry()
        wrapper = LatencyToolWrapper(tr, delay_ms=10)
        result = wrapper.execute("get_weather", {"city": "paris"})
        self.assertIn("18C", result)
        self.assertEqual(wrapper.call_count, 1)

    def test_properties(self):
        tr = ToolRegistry()
        wrapper = LatencyToolWrapper(tr, delay_ms=50)
        wrapper.execute("get_weather", {"city": "paris"})
        wrapper.execute("search_attractions", {"city": "rome"})
        self.assertEqual(wrapper.call_count, 2)
        self.assertEqual(wrapper.total_delay_s, 0.10)
