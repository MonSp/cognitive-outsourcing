import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CacheStats, DependencyAnalyzer, compute_cache_efficiency
from core.metrics import init_metrics, average_metrics


class TestCacheStats(unittest.TestCase):

    def test_initial_state(self):
        stats = CacheStats()
        self.assertEqual(stats.total_injected_tokens, 0)
        self.assertEqual(stats.total_eval_time, 0.0)
        self.assertEqual(stats.total_generate_time, 0.0)
        self.assertEqual(stats.injection_count, 0)
        self.assertEqual(stats.generate_count, 0)

    def test_cumulative_compute(self):
        stats = CacheStats(total_eval_time=1.5, total_generate_time=2.5)
        self.assertAlmostEqual(stats.cumulative_compute, 4.0)

    def test_avg_inject_time(self):
        stats = CacheStats(total_eval_time=1.0, injection_count=4)
        self.assertAlmostEqual(stats.avg_inject_time, 0.25)

    def test_avg_inject_time_zero(self):
        stats = CacheStats()
        self.assertEqual(stats.avg_inject_time, 0.0)

    def test_avg_generate_time(self):
        stats = CacheStats(total_generate_time=2.0, generate_count=5)
        self.assertAlmostEqual(stats.avg_generate_time, 0.4)

    def test_avg_generate_time_zero(self):
        stats = CacheStats()
        self.assertEqual(stats.avg_generate_time, 0.0)

    def test_summary(self):
        stats = CacheStats(
            total_injected_tokens=100,
            total_eval_time=1.0,
            total_generate_time=2.0,
            injection_count=5,
            generate_count=10,
            evict_count=1,
            compact_count=2,
            peak_cache_tokens=50,
        )
        summary = stats.summary()
        self.assertEqual(summary["total_injected_tokens"], 100)
        self.assertEqual(summary["injection_count"], 5)
        self.assertEqual(summary["generate_count"], 10)
        self.assertEqual(summary["evict_count"], 1)
        self.assertEqual(summary["compact_count"], 2)
        self.assertEqual(summary["peak_cache_tokens"], 50)
        self.assertAlmostEqual(summary["cumulative_compute_s"], 3.0)


class TestDependencyAnalyzer(unittest.TestCase):

    def test_independent_tools(self):
        analyzer = DependencyAnalyzer()
        analyzer.add_node("get_weather", {"city": "paris"})
        analyzer.add_node("get_weather", {"city": "london"})
        analyzer.add_node("get_weather", {"city": "rome"})

        groups, sequential = analyzer.classify()
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 3)
        self.assertEqual(len(sequential), 0)

    def test_sequential_tools(self):
        analyzer = DependencyAnalyzer()
        analyzer.add_node("search_flight", {"origin": "paris"})
        analyzer.add_node("book_flight", {"flight_id": "AF100"}, depends_on=[0])

        groups, sequential = analyzer.classify()
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(len(sequential), 1)

    def test_mixed_dependencies(self):
        analyzer = DependencyAnalyzer()
        analyzer.add_node("get_weather", {"city": "paris"})
        analyzer.add_node("get_weather", {"city": "london"})
        analyzer.add_node("plan_route", {"from": "paris", "to": "london"}, depends_on=[0, 1])

        groups, sequential = analyzer.classify()
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(sequential), 1)

    def test_recommend_batch_size_independent(self):
        analyzer = DependencyAnalyzer()
        analyzer.add_node("get_weather", {"city": "paris"})
        analyzer.add_node("get_weather", {"city": "london"})
        analyzer.add_node("get_weather", {"city": "rome"})
        analyzer.add_node("get_weather", {"city": "berlin"})

        self.assertEqual(analyzer.recommend_batch_size(max_batch=8), 4)
        self.assertEqual(analyzer.recommend_batch_size(max_batch=2), 2)

    def test_recommend_batch_size_sequential(self):
        analyzer = DependencyAnalyzer()
        analyzer.add_node("search", {"q": "test"})
        analyzer.add_node("fetch", {"url": "http://example.com"}, depends_on=[0])

        self.assertEqual(analyzer.recommend_batch_size(), 1)

    def test_recommend_batch_size_mixed(self):
        analyzer = DependencyAnalyzer()
        analyzer.add_node("get_weather", {"city": "paris"})
        analyzer.add_node("get_weather", {"city": "london"})
        analyzer.add_node("plan", {"data": "test"}, depends_on=[0, 1])

        self.assertEqual(analyzer.recommend_batch_size(), 2)

    def test_reset(self):
        analyzer = DependencyAnalyzer()
        analyzer.add_node("get_weather", {"city": "paris"})
        analyzer.reset()

        groups, sequential = analyzer.classify()
        self.assertEqual(len(groups), 0)
        self.assertEqual(len(sequential), 0)

    def test_node_index_tracking(self):
        analyzer = DependencyAnalyzer()
        idx0 = analyzer.add_node("tool_a", {})
        idx1 = analyzer.add_node("tool_b", {})
        idx2 = analyzer.add_node("tool_c", {})

        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 1)
        self.assertEqual(idx2, 2)


class TestComputeCacheEfficiency(unittest.TestCase):

    def test_basic_efficiency(self):
        metrics = {
            "cumulative_eval_time": 1.0,
            "cumulative_gen_time": 3.0,
            "total_prefill_tokens": 100,
            "total_gen_tokens": 50,
            "cache_injection_count": 5,
        }
        eff = compute_cache_efficiency(metrics)
        self.assertAlmostEqual(eff["injection_ratio"], 0.25)
        self.assertAlmostEqual(eff["generation_ratio"], 0.75)
        self.assertEqual(eff["tokens_per_injection"], 20.0)
        self.assertGreater(eff["compute_per_token_ms"], 0)

    def test_zero_metrics(self):
        metrics = init_metrics()
        eff = compute_cache_efficiency(metrics)
        self.assertEqual(eff["injection_ratio"], 0.0)
        self.assertEqual(eff["generation_ratio"], 0.0)
        self.assertEqual(eff["tokens_per_injection"], 0.0)
        self.assertEqual(eff["compute_per_token_ms"], 0.0)


class TestInitMetricsExtended(unittest.TestCase):

    def test_cumulative_fields(self):
        m = init_metrics()
        self.assertEqual(m["cumulative_eval_time"], 0.0)
        self.assertEqual(m["cumulative_gen_time"], 0.0)
        self.assertEqual(m["cumulative_compute_time"], 0.0)
        self.assertEqual(m["cache_injection_count"], 0)
        self.assertEqual(m["cache_generate_count"], 0)
        self.assertEqual(m["cache_evict_count"], 0)
        self.assertEqual(m["cache_compact_count"], 0)
        self.assertEqual(m["peak_cache_tokens"], 0)


class TestAverageMetricsExtended(unittest.TestCase):

    def test_cumulative_averaging(self):
        runs = [
            {"cumulative_compute_time": 1.0, "cache_injection_count": 5},
            {"cumulative_compute_time": 2.0, "cache_injection_count": 10},
        ]
        avg = average_metrics(runs)
        self.assertAlmostEqual(avg["cumulative_compute_time"], 1.5)
        self.assertAlmostEqual(avg["cache_injection_count"], 7.5)


if __name__ == "__main__":
    unittest.main()
