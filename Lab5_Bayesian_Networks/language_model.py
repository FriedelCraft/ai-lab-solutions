"""Count word transitions to build first- and second-order language models."""
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import random

START = "<START>"
END = "<END>"
CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


class GenerationLimit(RuntimeError):
    def __init__(self, partial_text):
        super().__init__("Length cap reached before END; not a completed sentence")
        self.partial_text = partial_text


class LanguageModel:
    def __init__(self, sentences, order=1):
        if order not in (1, 2):
            raise ValueError("Order must be 1 or 2")
        self.order = order
        self.counts = defaultdict(Counter)
        words = []
        for sentence in sentences:
            if isinstance(sentence, str):
                words.append(sentence.lower().split())
            else:
                words.append([word.lower() for word in sentence])
        if not words or any(not row for row in words):
            raise ValueError("Provide nonempty sentences")
        if any(w in (START.lower(), END.lower()) for row in words for w in row):
            raise ValueError("Boundary tokens are added by the model")
        self.vocabulary = sorted({w for row in words for w in row})
        # Use two START tokens when the context needs two previous words.
        for row in words:
            tokens = [START] * order + row + [END]
            for i in range(order, len(tokens)):
                context = tuple(tokens[i - order:i])
                self.counts[context][tokens[i]] += 1

    def distribution(self, context):
        if isinstance(context, str):
            context = (context,)
        context = tuple(context)
        if len(context) != self.order:
            raise ValueError(f"Need {self.order} previous tokens")
        # Looking up an unknown context should not add it to the counts.
        counts = self.counts.get(context)
        if not counts:
            return {}
        total = sum(counts.values())
        return {word: n / total for word, n in sorted(counts.items())}

    def next_word(self, context, mode="sample", rng=None):
        if mode not in ("sample", "greedy"):
            raise ValueError("Mode must be sample or greedy")
        probabilities = self.distribution(context)
        if not probabilities:
            raise ValueError(f"Unobserved context: {context}")
        if mode == "greedy":
            # Alphabetical tie break makes output reproducible.
            return min(probabilities, key=lambda w: (-probabilities[w], w))
        if rng is None:
            rng = random
        return rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]

    def generate(self, mode="sample", rng=None, max_words=50):
        context = (START,) * self.order
        generated = []
        # One additional draw allows END after exactly max_words words.
        for _ in range(max_words + 1):
            word = self.next_word(context, mode, rng)
            if word == END:
                return " ".join(generated)
            if len(generated) == max_words:
                break
            generated.append(word)
            context = context[1:] + (word,)
        raise GenerationLimit(" ".join(generated))

    def normalization_totals(self):
        return {" | ".join(context): sum(self.distribution(context).values())
                for context in sorted(self.counts)}

    def comparison_stats(self):
        # END stops a sentence, so it is not a prediction context.
        contexts = set(product(self.vocabulary, repeat=self.order))
        contexts.add((START,) * self.order)
        if self.order == 2:
            contexts.update((START, w) for w in self.vocabulary)
        outcomes = len(self.vocabulary) + 1  # END is a possible next token
        nonzero = sum(len(row) for row in self.counts.values())
        return {
            "candidate_contexts": len(contexts),
            "observed_contexts": len(self.counts),
            "unobserved_contexts": len(contexts - set(self.counts)),
            "dense_cpt_entries": len(contexts) * outcomes,
            "nonzero_cpt_entries": nonzero,
            "free_parameters_in_observed_rows": sum(len(row) - 1 for row in self.counts.values()),
        }


def main():
    root = Path(__file__).resolve().parent
    models = {
        "first_order": LanguageModel(CORPUS, 1),
        "second_order": LanguageModel(CORPUS, 2),
    }
    results = {"seed": 42, "corpus": CORPUS}
    for name, model in models.items():
        totals = model.normalization_totals()
        assert all(abs(total - 1) < 1e-12 for total in totals.values())
        rng = random.Random(42)
        samples = [model.generate(rng=rng) for _ in range(20)]
        greedy = []
        for _ in range(5):
            try:
                greedy.append(model.generate(mode="greedy"))
            except GenerationLimit as error:
                greedy.append(error.partial_text + " [TRUNCATED: no END]")
        comparison_samples = [model.generate(rng=random.Random(42 + i)) for i in range(5)]
        root.joinpath(f"generated_sentences_{name}.txt").write_text(
            "\n".join(f"{i}. {s}" for i, s in enumerate(samples, 1)) + "\n", encoding="utf-8")
        results[name] = {
            "cpt": {" | ".join(c): model.distribution(c) for c in sorted(model.counts)},
            "normalization_totals": totals,
            "predictions": {" | ".join(c): model.next_word(c, "greedy") for c in sorted(model.counts)},
            "stats": model.comparison_stats(), "samples_20": samples,
            "unique_sampled_sentences": len(set(samples)),
            "greedy_5": greedy, "sampled_5": comparison_samples,
        }

        print(f"\n{name.replace('_', ' ').title()}")
        print("Conditional probabilities:")
        for context in sorted(model.counts):
            print(context, "->", model.distribution(context))
        print("Normalization totals:", totals)
        print("\n20 sampled sentences:")
        for i, sentence in enumerate(samples, 1):
            print(f"{i}. {sentence}")
        print("\nFive greedy outputs:")
        for sentence in greedy:
            print(sentence)
        print("\nFive sampled outputs:")
        for sentence in comparison_samples:
            print(sentence)
        print("\nCPT counts:", results[name]["stats"])
        print("Distinct sentences among 20 samples:", len(set(samples)))
    text = json.dumps(results, indent=2)
    root.joinpath("results.json").write_text(text + "\n", encoding="utf-8")
    print("\nFull results saved to results.json")


if __name__ == "__main__":
    main()
