import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import mean_std, init_metrics, compute_metrics_table


class TestMeanStd(unittest.TestCase):

    def test_basic(self):
        m, s = mean_std([1.0, 2.0, 3.0])
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(s, 1.0)

    def test_single_value(self):
        m, s = mean_std([5.0])
        self.assertEqual(m, 5.0)
        self.assertEqual(s, 0.0)

    def test_empty(self):
        m, s = mean_std([])
        self.assertEqual(m, 0.0)
        self.assertEqual(s, 0.0)


class TestInitMetrics(unittest.TestCase):

    def test_fields(self):
        m = init_metrics()
        self.assertEqual(m["total_ttf"], 0.0)
        self.assertEqual(m["total_gen_time"], 0.0)
        self.assertEqual(m["total_prefill_time"], 0.0)
        self.assertEqual(m["total_gen_tokens"], 0)
        self.assertEqual(m["total_prefill_tokens"], 0)
        self.assertEqual(m["per_turn_ttf"], [])


class TestComputeMetricsTable(unittest.TestCase):

    def test_basic(self):
        sig = [
            {"total_ttf": 1.0, "quality_composite": 0.8},
            {"total_ttf": 1.2, "quality_composite": 0.75},
            {"total_ttf": 0.9, "quality_composite": 0.85},
        ]
        app = [
            {"total_ttf": 2.5, "quality_composite": 0.9},
            {"total_ttf": 2.6, "quality_composite": 0.92},
            {"total_ttf": 2.4, "quality_composite": 0.88},
        ]
        result = compute_metrics_table(sig, app, "Test")
        self.assertEqual(result["label"], "Test")
        self.assertGreater(result["speedup"], 2.0)
        self.assertLess(result["quality_delta"], 0)

    def test_zero_speedup_edge(self):
        sig = [{"total_ttf": 0.0, "quality_composite": 0.5}]
        app = [{"total_ttf": 2.0, "quality_composite": 0.5}]
        result = compute_metrics_table(sig, app)
        self.assertGreater(result["speedup"], 1000)
