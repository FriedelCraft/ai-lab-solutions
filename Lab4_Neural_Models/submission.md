# Lab 4: Neural Models

## Task 1: Problem understanding

Input space: `X = {0,1}^2`. Binary output space: `Y = {0,1}`, where 1 means the redundant sensors disagree.

| x1 | x2 | y |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```text
x2
1    (0,1):1 -------- (1,1):0
0    (0,0):0 -------- (1,0):1
     0                1       x1
```

The two classes occupy opposite diagonals; their convex hulls intersect at the square's center. One straight boundary cannot separate them. A single affine map followed by sigmoid therefore cannot classify all four points correctly. For this balanced full-batch dataset, the finite optimum of binary cross-entropy for affine logistic regression predicts 0.5 at each input, with loss ln(2), approximately 0.693147. This is a prediction about the linear model, not an additional experiment claimed to have been run.

**Representation claim:** Extra affine-only layers still compose to an affine map. Hidden nonlinearity, rather than layer count alone, makes the XOR decision representable.

## Task 2: Model design

The binary baseline is exactly **2 inputs -> 2 tanh hidden units -> 1 logit**. Its equations are `h=tanh(W1*x+b1)` and `z=W2*h+b2`. The reported probability is sigmoid(z).

`BCEWithLogitsLoss` computes binary cross-entropy with its sigmoid-related operations in a numerically stable form; logits are passed directly to it. Applying sigmoid before this loss would be incorrect. Full-batch Adam uses learning rate 0.03 and 4,000 updates on CPU.

The hidden nonlinearity permits a transformed representation where the final affine readout can distinguish disagreement. The hidden units have no supplied target values; backpropagation sends derivatives from the output loss through the hidden activation, and optimization adjusts their features to reduce that loss.

Successful learning requires: (1) final loss substantially below initial loss, (2) all four thresholded labels equal `[0,1,1,0]`, and (3) probabilities moving close to the desired labels. The first-layer gradient is inspected as a learning signal, but a nonzero gradient alone is insufficient evidence of success.

## Task 3: Implementation specification and corrections

The binary specification guiding code generation was:

> Implement the four XOR examples using PyTorch, exactly two inputs, two hidden tanh units, and one raw output logit with BCEWithLogitsLoss. Use seeded random initialization and full-batch CPU Adam for 4,000 steps at learning rate 0.03. Report initial/final losses, all probabilities, thresholded labels, and the first-layer gradient after backward. Repeat with all parameters zero and record both hidden-weight rows over several steps. Compare sigmoid, tanh, and ReLU while keeping seed and all other training settings identical. Do not enlarge the model to hide optimization failure.

The final code is [neural_xor.py](neural_xor.py). In `train`, `model(X)` performs the forward pass; `criterion(logits,target)` forms a mean scalar loss; `loss.backward()` performs reverse-mode differentiation; `optimizer.step()` updates parameters. `optimizer.zero_grad()` prevents accumulation between updates.

The first attempt used seed 7 and did not classify all four XOR examples correctly. The final run changes only the seed to 0, an allowed engineering adjustment; architecture, task, optimizer, learning rate, and update count are preserved. The same seed 0 is used for all three activation runs. A final forward/backward pass after the last update ensures the reported final gradient belongs to the reported final parameters. These are documented implementation/measurement decisions, rather than claims about edits personally made by the student.

**Inspection versus execution:** Architecture, target shapes, loss choice, and the presence of backward/step can be checked in source. Whether loss decreases, XOR is learned, hidden rows stay equal, or an activation yields useful gradients requires execution and measurement.

## Task 4A: Basic learning check

Baseline: tanh, CPU, PyTorch 2.14.1+cpu, seed 0, Adam, learning rate 0.03, 4,000 steps.

- Initial binary loss: **0.71515930**.
- Final binary loss: **0.0000860917**.
- Final predictions: **[0,1,1,0]**, four correct.

| Input | Target | Final probability | Prediction |
| --- | --- | --- | --- |
| (0,0) | 0 | 0.0000620772 | 0 |
| (0,1) | 1 | 0.99989343 | 1 |
| (1,0) | 1 | 0.99987757 | 1 |
| (1,1) | 0 | 0.0000530218 | 0 |

Thresholding uses `p >= 0.5`. The decline in loss together with all four correct labels supports successful learning on this complete tiny input space.

## Task 4B: Backpropagation check

The first-layer gradient before the first optimizer update is:

```text
[[ 0.0005053986,  0.0006069484],
 [-0.0425690413, -0.0447729230]]
```

Its Euclidean/Frobenius norm is **0.06178480**. After training, a fresh backward pass gives:

```text
[[ 0.000003529361, -0.000004317953],
 [-0.000005872130,  0.000003378786]]
```

`model.hidden.weight.grad` is the matrix of partial derivatives `dL/dW1`. Since the scalar objective is `L=(1/4)*sum_i L_i`, differentiation gives `(1/4)*sum_i dL_i/dW1`. A small final gradient is consistent with low remaining error here; it is not itself a failure. Adam uses an adaptive transformation of these gradients, rather than directly subtracting the raw gradient with a fixed step.

## Task 4C: Symmetry experiment

All weights **and biases** are set to zero so the hidden units begin identical. Tanh, architecture, optimizer, and training duration remain unchanged.

| Before update | Hidden weight row 1 | Hidden weight row 2 | Equal? |
| --- | --- | --- | --- |
| 0 | [0,0] | [0,0] | Yes |
| 1 | [0,0] | [0,0] | Yes |
| 10 | [0,0] | [0,0] | Yes |
| 100 | [0,0] | [0,0] | Yes |
| 3999 | [0,0] | [0,0] | Yes |

Initial and final loss are **0.69314718**. All probabilities are 0.5; the chosen threshold predicts `[1,1,1,1]`, only two correct. First-layer gradients remain zero.

Identical hidden units and equal outgoing weights yield the same outputs, incoming gradients, and updates, preserving symmetry. In this particular all-zero tanh network, hidden outputs and output weights are zero, while the balanced labels cancel the mean output-bias gradient. The network is stationary, a stronger failure than merely learning duplicate features. Identical hidden weights alone do not guarantee equal gradients if biases or outgoing weights differ; the experiment controls all of these.

## Task 4D: Activation comparison

Only the hidden activation changes. All three networks start from the same random parameter values and use the same data and optimizer settings. The early gradient is measured before update 0.

| Hidden activation | Initial loss | Final loss | Correct inputs | Early first-layer gradient norm |
| --- | --- | --- | --- | --- |
| Sigmoid | 0.69765902 | 0.0002630014 | 4/4 | 0.0008908799 |
| Tanh | 0.71515930 | 0.0000860917 | 4/4 | 0.0617848039 |
| ReLU | 0.70688516 | 0.4773875475 | 3/4 | 0.0016988000 |

Sigmoid's derivative is at most 1/4, whereas tanh's is at most 1; this helps explain different gradient magnitudes but does not by itself establish the cause of every optimization outcome. Sigmoid successfully learns in this run despite its small initial gradient. ReLU's second hidden-weight row has zero initial gradient; this run converges to probabilities approximately `[0.666664,0.666664,0.666664,0.000007734]`, misclassifying (0,0). A larger tanh gradient and a failed ReLU fit are observations for this seed and configuration, not universal rankings.

**Saturation versus inactive ReLU:** Inspect preactivations and activations. Sigmoid saturation has large-magnitude preactivations and outputs near 0 or 1, with small but generally nonzero derivatives. An inactive ReLU has nonpositive preactivation, output zero, and derivative zero for negative input. Both can yield small parameter gradients, but through different mechanisms.

## Task 5: Three-class extension

Targets change to `[0,1,1,2]`: both inactive, disagree, both active. The hidden layer remains two tanh units; output changes to `Linear(2,3)` with three raw logits and `CrossEntropyLoss`. In PyTorch the output weight shape is **(3,2)** and bias shape is (3,); each example has three logits, giving batch shape (4,3).

The extension specification was:

> Keep the two-input/two-tanh-hidden representation and training settings. Replace the binary output with three logits and use CrossEntropyLoss with integer targets [0,1,1,2]. Report the four softmax vectors, predicted classes, their sums, and the effect of adding 100 to every logit. Pass raw logits to the loss.

Initial multiclass loss is **1.05910838**; final loss is **0.0000433912**.

| Input | Target | P(class 0) | P(class 1) | P(class 2) | Prediction |
| --- | --- | --- | --- | --- | --- |
| (0,0) | 0 | 0.99994409 | 0.0000558570 | 0.0000000209 | 0 |
| (0,1) | 1 | 0.0000267971 | 0.99996459 | 0.0000086137 | 1 |
| (1,0) | 1 | 0.0000261495 | 0.99996614 | 0.0000077325 | 1 |
| (1,1) | 2 | 0.0000006174 | 0.0000477363 | 0.99995160 | 2 |

The four sums are `[0.99999994,1.0,1.0,0.99999994]`. In particular, the first row sums to approximately one. Softmax normalizes each `exp(z_k)` by the sum of all exponentials, so the theoretical sum is exactly one.

For one example with one-hot target y, `L=-sum_k y_k*log(p_k)`. Differentiating log-softmax yields `dL/dz_k=p_k-y_k`. With this mean four-example objective, the derivative for each example is `(p-y)/4` before accumulating parameter gradients. PyTorch `CrossEntropyLoss` expects raw logits and includes stable log-softmax internally.

Adding 100 to every logit changes probabilities by at most **2.1464e-10** in this run. The common factor `exp(100)` cancels algebraically. Subtracting the maximum logit exploits the same invariance while preventing overflow; very small probabilities can still underflow numerically.

**Connection to next-token prediction:** Logits, softmax, categorical cross-entropy, and the p-y derivative retain their form for a large vocabulary. The representation, context processing, output matrix size, computational cost, and training data scale change substantially.

## Reflection questions

1. **Depth versus nonlinearity:** Affine depth collapses to one affine map. XOR requires nonlinear representation; two nonlinear hidden units can supply it.
2. **Useful learning signal:** The tanh loss fell from 0.71515930 to 0.0000860917 and all four probabilities approached their correct labels. That combines gradient evidence with actual improvement, rather than treating nonzero derivatives as sufficient.
3. **Initialization symmetry:** The controlled all-zero initialization preserves identical units and, here, yields zero gradients. Random initialization allows units to begin with distinct features and receive different updates.
4. **Activation effects:** The measured gradient norms differ by activation. Their derivatives supply a mathematical explanation for possible shrinkage or inactivity, but the table only establishes outcomes for this experiment; sigmoid succeeds and ReLU fails here.
5. **Output/loss pairing:** Binary prediction is Bernoulli and uses one logit with binary cross-entropy. Mutually exclusive three-class prediction is categorical and uses three logits with categorical cross-entropy. Loss APIs must receive the representation they expect.
6. **Productivity and verification:** AI assistance generated the training loop and result collection. Verification caught unsuccessful seed-7 optimization and ensured the reported gradient came from the final parameters. Human review remains necessary for model assumptions and interpretation.
7. **Scaling tests:** Keep loss monitoring, validation accuracy, sampled predictions, probability normalization, and selected gradient norms. Exhaustive input checks and finite-difference checks of every parameter become too expensive; use small controlled cases or selected parameters instead.

Full-precision measured data are in [results.json](results.json). Code and analysis were prepared with Codex assistance; no unexecuted experimental result is presented as measured.
