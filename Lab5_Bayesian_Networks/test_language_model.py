"""Checks for normalization, context conditioning, boundaries, and generation.

Run: python -m unittest discover -s Lab5_Bayesian_Networks -v
"""
import random
import unittest
from language_model import LanguageModel, GenerationLimit, CORPUS, START, END


class LanguageModelTests(unittest.TestCase):
    def test_probability_rows(self):
        for order in [1, 2]:
            model = LanguageModel(CORPUS, order)
            for context in model.counts:
                row = model.distribution(context)
                self.assertAlmostEqual(sum(row.values()), 1)
                self.assertTrue(all(0 <= p <= 1 for p in row.values()))

    def test_exact_first_order_counts(self):
        model = LanguageModel(CORPUS)
        self.assertEqual(model.counts[("the",)], {"cat": 3, "dog": 3, "mat": 2, "rug": 2, "park": 2})
        self.assertEqual(model.distribution("cat"), {"ran": 1/3, "sat": 2/3})
        self.assertEqual(model.distribution("the")["cat"], 1/4)

    def test_no_cross_sentence_transitions(self):
        model = LanguageModel(CORPUS)
        self.assertNotIn((END,), model.counts)
        self.assertEqual(model.distribution(START), {"the": 1.0})
        self.assertEqual(model.distribution("mat"), {END: 1.0})

    def test_second_order_uses_both_parents(self):
        model = LanguageModel(CORPUS, 2)
        self.assertEqual(model.distribution((START, "the")), {"cat": 0.5, "dog": 0.5})
        self.assertEqual(model.distribution(("on", "the")), {"mat": 0.5, "rug": 0.5})
        self.assertEqual(model.distribution(("to", "the")), {"park": 1.0})

    def test_unobserved_context_is_explicit(self):
        model = LanguageModel(CORPUS)
        original = len(model.counts)
        self.assertEqual(model.distribution("unicorn"), {})
        with self.assertRaises(ValueError):
            model.next_word("unicorn")
        self.assertEqual(len(model.counts), original)

    def test_seeded_generation_and_greedy(self):
        for order in [1, 2]:
            model = LanguageModel(CORPUS, order)
            r1, r2 = random.Random(42), random.Random(42)
            samples = [model.generate(rng=r1) for _ in range(20)]
            self.assertEqual(samples, [model.generate(rng=r2) for _ in range(20)])
            self.assertTrue(all(START not in s and END not in s for s in samples))
            self.assertTrue(all(w in model.vocabulary for s in samples for w in s.split()))
            self.assertGreater(len(set(samples)), 1)
            if order == 1:
                # Greedy follows the same cycle forever on this corpus.
                with self.assertRaises(GenerationLimit) as caught:
                    model.generate("greedy")
                self.assertTrue(caught.exception.partial_text.startswith("the cat sat on the cat"))
            else:
                self.assertEqual(len({model.generate("greedy") for _ in range(5)}), 1)

    def test_length_cap_is_not_silent_completion(self):
        with self.assertRaises(RuntimeError):
            LanguageModel(CORPUS).generate("greedy", max_words=1)


if __name__ == "__main__":
    unittest.main()
