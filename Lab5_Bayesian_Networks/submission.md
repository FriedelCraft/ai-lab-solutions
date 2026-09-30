# Lab 5: Bayesian Networks and Autoregressive Language Models

## Parts I-II: Probability, language, and dependencies

**Question 1: Why is autoregressive decomposition useful for generation?**

The probability chain rule writes a joint distribution as a product of next-token conditionals:

`P(x1,...,xT) = P(x1) * product_{t=2..T} P(xt | x1,...,x(t-1))`.

This permits sequential generation: sample the first token, then repeatedly sample a token conditional on the generated prefix. There is no need to enumerate every possible sentence. With boundary markers, the first distribution is conditioned on START and END denotes termination. This factorization is exact before any context-window approximation is imposed.

**Question 2: What independence assumption does the first-order network make?**

For t >= 3, `Xt` is conditionally independent of `X1,...,X(t-2)` given `X(t-1)`:

`P(Xt | X1,...,X(t-1)) = P(Xt | X(t-1))` in the model.

The graph is a chain `X1 -> X2 -> ... -> XT`; its joint factorization is `P(X1)*product P(Xt|X(t-1))`. The independence assumption approximates real language, which depends on much more context.

## Parts III-IV: Corpus and conditional probability tables

The exact six-sentence starting corpus is used, without adding training text:

```text
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
```

Ordinary words are lowercased and split on whitespace. `<START>` and `<END>` are reserved boundary markers. Each sentence is counted separately, so END is never linked to the next sentence's START. The ordinary vocabulary has ten words.

**Question 3: Construct P(next | current) and identify zero transitions.**

For an observed context w, `P(v|w) = C(w,v)/sum_u C(w,u)`. The denominator counts successors of that specific context, not all corpus transitions.

| Current word | Successor counts | Conditional probabilities | Greedy prediction |
| --- | --- | --- | --- |
| the | cat:3, dog:3, mat:2, rug:2, park:2 | cat:1/4, dog:1/4, mat:1/6, rug:1/6, park:1/6 | cat (tie with dog) |
| cat | sat:2, ran:1 | sat:2/3, ran:1/3 | sat |
| dog | sat:2, ran:1 | sat:2/3, ran:1/3 | sat |
| sat | on:4 | on:1 | on |
| ran | to:2 | to:1 | to |

The word `the` occurs in the sentence beginning and before mat/rug/park; **all twelve outgoing occurrences** are included. Counting only its sentence-initial occurrences would produce the wrong CPT for the first-order model.

The remaining observed rows are START -> the, on -> the, to -> the, and mat/rug/park -> END, each with probability 1. Every unlisted transition from an observed row has probability zero under this unsmoothed maximum-likelihood model. For example, P(the|the)=0, P(on|cat)=0, P(ran|sat)=0, and P(END|ran)=0. Zero here means unobserved in this corpus, not impossible in all language. END is terminal; it has no outgoing prediction row.

## Parts V-VI: Implementation and inspection

The implementation specification guiding first-order generation was:

> Use only ordinary Python data structures and random sampling to implement a first-order language model over the six supplied sentences. Lowercase words, add START and END per sentence, count within-sentence transitions, normalize each context's successor counts, display selected CPT rows, and predict argmax with a deterministic tie rule. Support greedy and weighted-sampling generation. Stop on END and distinguish a length-capped prefix from a completed sentence. Save 20 sampled sentences with a fixed seed and run probability/count/boundary tests. Report unseen contexts explicitly rather than inventing a distribution.

The final implementation is [language_model.py](language_model.py), with [test_language_model.py](test_language_model.py) containing standard-library tests. There is no pretrained model or machine-learning library in this lab.

**Question 4: Where are transition counts stored?** `LanguageModel.counts` is a `defaultdict(Counter)`. In the first-order model its keys are one-token tuples such as `('the',)`, and its values count each successor.

**Question 5: Where are probabilities computed?** `distribution(context)` divides each successor count by `sum(counts.values())` for the selected context. Rows are sorted for reproducible output and tie handling. Unobserved outcomes are omitted from the sparse dictionary and have zero probability in an observed row.

**Question 6: How is the next word chosen?** `next_word(...,mode='greedy')` chooses an argmax, with alphabetic order breaking ties. Sampling uses `random.choices` with the CPT probabilities as weights and a seeded random generator. Greedy repeats the same choices for a fixed prefix, while sampling selects different supported words according to their probabilities.

**Question 7: What happens for a word with no observed transition?** `distribution` returns `{}` and `next_word` raises `ValueError`. An unseen conditioning event has no estimated conditional distribution here; it is not a valid all-zero row. No smoothing or invented fallback has been added. `counts.get(context)` prevents an unseen query from silently creating an empty row in the count table.

## Part VII: Probability-model testing

The normalization condition applies to every **observed, nonterminal conditioning row**. The implementation checks nonnegative bounded probabilities and totals within numerical tolerance of one.

| First-order context | Row total |
| --- | --- |
| START | 1.0000000000 |
| cat | 1.0000000000 |
| dog | 1.0000000000 |
| mat | 1.0000000000 |
| on | 1.0000000000 |
| park | 1.0000000000 |
| ran | 1.0000000000 |
| rug | 1.0000000000 |
| sat | 1.0000000000 |
| the | approximately 1.0000000000 |
| to | 1.0000000000 |

All fifteen second-order observed rows also sum to approximately one. Seven executed tests pass:

1. Every observed row has probabilities in [0,1] and sum approximately one.
2. Exact `the` and `cat` counts/probabilities match manual calculation.
3. START/END are handled separately within sentences, without cross-sentence transitions.
4. Second-order predictions change with both context words.
5. An unknown context is explicit and does not mutate counts.
6. Sampling is reproducible, produces valid vocabulary words, and varies; greedy behavior matches the model.
7. A length cap raises an explicit incomplete-generation condition.

Run from the solution root with `python -m unittest discover -s Lab5_Bayesian_Networks -v`. The observed result is `Ran 7 tests ... OK`.

**Question 8: What if a row total is 0.87?** It is not a valid normalized distribution over the stated full outcome set: 0.13 of probability mass is missing. This is far larger than normal floating-point rounding. Check whether an outcome was omitted, the wrong denominator was used, or display filtering hid entries. If the sum was over only a selected subset, it does not test the full-row invariant.

## Part VIII: Predicting the next word

The five distributions and predictions requested by the lab are in the Question 3 table. `the` has two equally most probable successors; the implementation consistently reports cat. Prediction reflects corpus-relative transition frequency, not general English knowledge.

**Question 9: Do predictions always match human expectations?** No. After a prefix such as "the cat sat on the", a person expects mat or rug. The first-order model sees only `the` and still assigns cat/dog a combined probability of 1/2. Human expectations use the longer prefix, semantics, and broader experience. The mismatch identifies limited context and data, rather than an arithmetic error in this CPT.

## Parts IX-X: Generated text and decoding modes

Twenty sampled first-order sentences are saved in [generated_sentences_first_order.txt](generated_sentences_first_order.txt); twenty second-order samples are saved in [generated_sentences_second_order.txt](generated_sentences_second_order.txt). Each set uses a continuous `random.Random(42)` stream. All forty saved samples reached END before the 50-word cap; START/END are omitted from displayed sentence text.

First-order examples from the saved 20:

```text
the cat ran to the park
the dog ran to the cat ran to the cat sat on the park
the cat sat on the cat sat on the park
the dog sat on the rug
the park
```

For the five-sentence mode comparison, sampled run i uses seed `42+i`, with i starting at zero. Its first-order outputs are:

```text
1. the cat ran to the park
2. the park
3. the mat
4. the dog ran to the cat ran to the park
5. the dog sat on the mat
```

All five first-order greedy outputs are identical 50-word prefixes, clearly marked `[TRUNCATED: no END]` in `results.json`. They begin:

```text
the cat sat on the cat sat on the cat sat on the ...
```

The cycle is `the -> cat -> sat -> on -> the`. At each visit to `the`, the greedy choice is cat, so no terminal noun is selected and END is never reached. A length cap prevents an infinite run but does not turn its prefix into a completed sentence. This is an outcome of the specified model and decoding rule, not a reason to change probabilities to force a nicer result.

**Question 10: Which mode varies more, and why?** Sampling produced five distinct texts in this comparison, while greedy produced one repeated prefix. Sampling draws supported alternatives with their assigned probabilities; greedy always makes the same local argmax choices. Sampling also has a positive probability of selecting mat/rug/park after `the` and then terminating. Diversity is a result of these finite samples, not a claim of unlimited or guaranteed uniqueness.

## Parts XI-XII: Second-order Bayesian network

The second-order assumption is `P(Xt | X1,...,X(t-1)) = P(Xt | X(t-2),X(t-1))` for t >= 3. Each token then has edges from the preceding two tokens. For four variables:

`P(X1,X2,X3,X4) = P(X1) * P(X2|X1) * P(X3|X1,X2) * P(X4|X2,X3)`.

The graph retains the chain edges and adds skip edges such as X1 -> X3 and X2 -> X4. Generation samples from the corresponding conditional distributions in sequence.

**Question 11: What changes?** The graph gives each later token two parents rather than one. CPT rows are indexed by ordered pairs rather than individual words. Prediction can distinguish `on the` from `to the`, unlike the first-order model. More data are needed because observations are distributed across many more contexts.

The second-order implementation specification was:

> Extend the same count-based model to estimate P(next|previous two). Store observed triple counts using ordered-pair context keys. Pad each sentence with two START tokens to define its first-token and second-token distributions, and stop on END. Preserve the original corpus, normalization, greedy/sampling behavior, explicit unseen-context handling, and reproducible generation. Compare both models using clearly defined CPT/context counts and saved sample examples.

Two START tokens encode the initial distribution P(X1|START,START) and the first transition P(X2|START,X1); later rows use the actual preceding word pair. This convention handles early tokens without crossing sentence boundaries or introducing another model.

All observed second-order rows are:

| Context | Distribution |
| --- | --- |
| START,START | the:1 |
| START,the | cat:1/2, dog:1/2 |
| the,cat | sat:2/3, ran:1/3 |
| the,dog | sat:2/3, ran:1/3 |
| cat,sat | on:1 |
| dog,sat | on:1 |
| cat,ran | to:1 |
| dog,ran | to:1 |
| sat,on | the:1 |
| ran,to | the:1 |
| on,the | mat:1/2, rug:1/2 |
| to,the | park:1 |
| the,mat | END:1 |
| the,rug | END:1 |
| the,park | END:1 |

## Part XIII: Comparing the models

To make parameter counts reproducible, candidate histories include every ordinary-word history, the all-START initial history, and, for second order, `(START,word)` histories. END is excluded from histories because it terminates generation. There are ten ordinary words and eleven possible successor tokens including END. Counts below describe a stationary, shared CPT, rather than separate tables for each sentence position.

| Measure | First order | Second order |
| --- | --- | --- |
| Candidate contexts under the stated convention | 11 | 111 |
| Observed context rows | 11 | 15 |
| Unobserved contexts | 0 | 96 |
| Entries in a complete dense CPT | 121 | 1,221 |
| Nonzero stored probability entries | 17 | 19 |
| Independent probabilities in observed support (sum of row-support-size minus 1) | 6 | 4 |
| Unique sentences in 20 sampled generations | 16 | 6 |
| Typical sampled coherence | Fragmentation and repeated clauses | Matches the six training sentence patterns |

The lab's "zero-probability contexts" is interpreted here as **unobserved contexts** with zero empirical count. Their conditional distributions are undefined under this unsmoothed estimator, not normalized all-zero rows. Within observed rows, unobserved successor transitions genuinely have estimated probability zero. Dense entries, sparse nonzero entries, and independent parameters are different counts; the table labels each explicitly.

Five second-order sampled outputs, using the same comparison seeds as first order, are:

```text
1. the cat ran to the park
2. the dog ran to the park
3. the dog sat on the mat
4. the cat ran to the park
5. the cat sat on the rug
```

All five second-order greedy outputs are `the cat sat on the mat`. The second-order model distinguishes initial `START,the`, locative `on,the`, and directional `to,the`. It therefore prevents some first-order recombinations. On this tiny corpus it generates only the six training patterns, so improved local coherence comes with less observed diversity. This does not establish general natural-language understanding.

**Question 12: Why can more context help and hurt?** More history can disambiguate the next token, as `on the` versus `to the` demonstrates. With V ordinary words, word-only conditioning histories grow from V to V^2 and a dense successor table from roughly V^2 to V^3 entries. Limited data leave many pair contexts unseen or estimated from very few examples. The four free parameters on this observed sparse support do not contradict dense growth: most of this tiny second-order model's observed rows are deterministic, and most candidate rows are not estimable at all.

## Part XIV: Connection to modern language models

Both count-based and neural autoregressive models factorize a sequence into next-token conditionals and generate by sequential sampling or another decoder. Here explicit CPTs are estimated by counting with a fixed one/two-token window. A modern neural model computes a distribution from a much longer prefix using learned parameters, typically producing vocabulary logits followed by softmax. Its architecture and data scale differ substantially, but the underlying next-token probability objective remains the same.

## Part XV: LLM assistance and reflection

**Question 13: Why is a precise specification preferable?** "Write a language model" leaves representation, context size, estimation, and decoding unspecified. The explicit conditional/counting specification states the intended model and rules out an irrelevant pretrained neural system. It also supplies measurable invariants: exact context counts, row normalization, boundary behavior, and context dependence. An implementation is only one realization of that model; a correctly named class does not demonstrate that the mathematical conditional has been implemented.

Codex generated the implementation and tests. Inspected code includes the context-specific denominator in `distribution`, the per-sentence count loop, and `next_word`'s mode separation. The initial runner assumed greedy generation would finish and failed when the first-order cycle reached the cap. It was corrected to catch `GenerationLimit`, preserve the partial text, and label it explicitly incomplete. The generation probabilities and greedy rule were left intact. This is a real correction made during preparation and then verified by the updated tests.

The executed implementation tests support model consistency. Student review is still required for the conceptual interpretation and personal understanding; no fabricated account of an earlier student LLM session is included.

## Final question

**Question 14: What did viewing the model as a Bayesian network add?** It made dependencies explicit: graph edges show which previous tokens can influence a prediction. It gave a factorization of the joint probability and a principled generation process through sequential conditional sampling. It exposed the independence assumptions that cause first-order mistakes, and showed how adding a parent changes the CPT and increases sparsity. Finally, it supplied tests derived from the probabilistic model, including normalization and context sensitivity, so the generated program could be checked against more than plausible-looking output.

## Deliverables

- First- and second-order implementations: the same `LanguageModel` class with `order=1` or `order=2`.
- Selected and complete observed CPTs: tables above and full-precision [results.json](results.json).
- Generated samples: the two linked 20-sentence text files; five-per-mode examples are in `results.json` and discussed above.
- Normalization and other checks: seven passing tests, with every observed row total in `results.json`.
- Answers to Questions 1-14 and the actual inspected/corrected-code reflection: this report.
