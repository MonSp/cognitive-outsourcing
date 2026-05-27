import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quality import (
    KitchenQualityEvaluator, SemanticScorer, _keyword_hit, _semantic_hit
)


class TestSemanticScorer(unittest.TestCase):

    def test_exact_match(self):
        scorer = SemanticScorer()
        score = scorer.similarity("spaghetti_bolognese chicken_stir_fry",
                                   "spaghetti_bolognese chicken_stir_fry")
        self.assertGreaterEqual(score, 0.99)

    def test_underscore_vs_space(self):
        scorer = SemanticScorer()
        score = scorer.similarity("spaghetti_bolognese", "spaghetti bolognese")
        self.assertGreaterEqual(score, 0.5)

    def test_partial_match(self):
        scorer = SemanticScorer()
        score = scorer.similarity("chicken_stir_fry caprese_salad omelette",
                                   "I made chicken stir fry and salad")
        self.assertGreater(score, 0.1)

    def test_no_match(self):
        scorer = SemanticScorer()
        score = scorer.similarity("spaghetti_bolognese", "completely unrelated text")
        self.assertLess(score, 0.5)

    def test_empty_inputs(self):
        scorer = SemanticScorer()
        self.assertEqual(scorer.similarity("", "text"), 0.0)
        self.assertEqual(scorer.similarity("text", ""), 0.0)
        self.assertEqual(scorer.similarity("", ""), 0.0)


class TestKeywordHit(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(_keyword_hit(["paris", "rome"], "I visited Paris and Rome"), 1.0)
        self.assertEqual(_keyword_hit(["paris", "rome"], "I visited Paris"), 0.5)
        self.assertEqual(_keyword_hit(["paris"], "I visited London"), 0.0)

    def test_empty_keywords(self):
        self.assertEqual(_keyword_hit([], "some text"), 0.0)


class TestSemanticHit(unittest.TestCase):

    def test_fallback(self):
        score = _semantic_hit(["paris", "rome"], "Paris and Rome are beautiful")
        self.assertGreater(score, 0.5)

    def test_with_scorer(self):
        scorer = SemanticScorer()
        score = _semantic_hit(["spaghetti_bolognese"], "I made spaghetti bolognese", scorer)
        self.assertGreaterEqual(score, 0.3)


class TestKitchenQualityEvaluator(unittest.TestCase):

    def test_recipe_mention(self):
        gt = {"expected_recipes": ["spaghetti_bolognese", "chicken_stir_fry"]}
        evaluator = KitchenQualityEvaluator(gt)
        scores = evaluator.evaluate("I made spaghetti bolognese and chicken stir fry", [], [])
        self.assertGreaterEqual(scores["recipe_mentioned"], 0.35)

    def test_allergen_aware(self):
        gt = {"allergens": ["dairy"], "forbidden_foods": ["cheese_parmesan"]}
        evaluator = KitchenQualityEvaluator(gt)
        scores = evaluator.evaluate(
            "This dish is dairy-free and avoids all cheese products", [], [])
        self.assertGreaterEqual(scores["allergen_aware"], 0.5)

    def test_shopping_list(self):
        gt = {"shopping_items": ["tomato", "garlic", "basil"]}
        evaluator = KitchenQualityEvaluator(gt)
        scores = evaluator.evaluate(
            "Shopping list: tomato, garlic, basil, and oregano", [], [])
        self.assertGreater(scores["shopping_list_items"], 0.5)

    def test_tool_execution_rate(self):
        gt = {"expected_tool_sequence": ["get_recipe", "check_ingredients", "set_oven"]}
        evaluator = KitchenQualityEvaluator(gt)
        tool_log = [
            {"tool": "get_recipe", "args": {}},
            {"tool": "check_ingredients", "args": {}},
        ]
        scores = evaluator.evaluate("summary", tool_log, [])
        self.assertEqual(scores["tool_execution_rate"], 2.0 / 3.0)

    def test_composite_score_range(self):
        gt = {
            "expected_recipes": ["spaghetti_bolognese"],
            "shopping_items": ["tomato"],
            "allergens": ["dairy"],
            "forbidden_foods": [],
            "expected_tool_sequence": ["get_recipe"],
        }
        evaluator = KitchenQualityEvaluator(gt)
        tool_log = [{"tool": "get_recipe"}]
        scores = evaluator.evaluate(
            "I made spaghetti bolognese. Shopping: tomato. Dairy-aware.", tool_log, [])
        self.assertTrue(0.0 <= scores["composite"] <= 1.0)
