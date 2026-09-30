"""Train the XOR network, then try the three-class sensor task."""
import json
from pathlib import Path
import torch
from torch import nn

SEED = 0
STEPS = 4000
LR = 0.03
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])
Y_THREE = torch.tensor([0, 1, 1, 2], dtype=torch.long)


class SensorNet(nn.Module):
    def __init__(self, activation="tanh", outputs=1):
        super().__init__()
        self.hidden = nn.Linear(2, 2)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError("Use sigmoid, tanh, or relu")
        self.output = nn.Linear(2, outputs)

    def forward(self, x):
        hidden = self.activation(self.hidden(x))
        return self.output(hidden)  # The loss functions take raw logits.


def train(activation="tanh", outputs=1, zero=False, seed=SEED, steps=STEPS):
    torch.manual_seed(seed)
    model = SensorNet(activation, outputs)
    if zero:
        # Zero the biases as well so both hidden units start the same.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
    if outputs == 1:
        criterion = nn.BCEWithLogitsLoss()
        target = Y
    else:
        criterion = nn.CrossEntropyLoss()
        target = Y_THREE
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    initial_loss = criterion(model(X), target).item()
    early_grad = None
    snapshots = []
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, target)
        loss.backward()
        if step == 0:
            early_grad = model.hidden.weight.grad.detach().clone()
        if zero and step in [0, 1, 10, 100, steps - 1]:
            weights = model.hidden.weight.detach()
            snapshots.append({
                "before_update": step,
                "weights": weights.tolist(),
                "identical_rows": torch.equal(weights[0], weights[1]),
            })
        optimizer.step()
    # Recompute the gradient after the last update, at the final weights.
    optimizer.zero_grad()
    final_logits = model(X)
    final_loss = criterion(final_logits, target)
    final_loss.backward()
    with torch.no_grad():
        if outputs == 1:
            probs = torch.sigmoid(final_logits)
            predictions = (probs[:, 0] >= 0.5).long()
            expected = Y[:, 0].long()
        else:
            probs = final_logits.softmax(dim=1)
            predictions = probs.argmax(dim=1)
            expected = Y_THREE
        result = {
            "activation": activation, "seed": seed, "steps": steps, "lr": LR,
            "initial_loss": initial_loss, "final_loss": final_loss.item(),
            "probabilities": probs.tolist(), "predictions": predictions.tolist(),
            "correct": int((predictions == expected).sum().item()),
            "early_gradient": early_grad.tolist(),
            "early_gradient_norm": early_grad.norm().item(),
            "final_gradient": model.hidden.weight.grad.tolist(),
        }
        if zero:
            result["symmetry_snapshots"] = snapshots
        if outputs == 3:
            shifted_probs = (final_logits + 100).softmax(dim=1)
            result["probability_sums"] = probs.sum(dim=1).tolist()
            result["softmax_shift_max_difference"] = (probs - shifted_probs).abs().max().item()
            assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
            assert torch.allclose(probs, shifted_probs, atol=1e-5)
    return result


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    experiments = {name: train(name) for name in ["sigmoid", "tanh", "relu"]}
    zero = train("tanh", zero=True)
    three = train("tanh", outputs=3)
    assert experiments["tanh"]["correct"] == 4, "Baseline did not learn all XOR labels"
    assert all(row["identical_rows"] for row in zero["symmetry_snapshots"])
    assert three["correct"] == 4, "Three-class extension did not learn all classes"
    data = {"python_library": "PyTorch", "torch_version": torch.__version__,
            "device": "CPU", "binary": experiments, "zero_initialization": zero,
            "three_class": three}
    text = json.dumps(data, indent=2)
    Path(__file__).with_name("results.json").write_text(text + "\n", encoding="utf-8")

    print("Activation comparison (same seed and training settings)")
    print(f"{'Activation':<12} {'Initial loss':>13} {'Final loss':>13} {'Correct':>9} {'Early grad norm':>17}")
    for name, result in experiments.items():
        print(f"{name:<12} {result['initial_loss']:13.8f} {result['final_loss']:13.8f} "
              f"{result['correct']:7d}/4 {result['early_gradient_norm']:17.8f}")

    baseline = experiments["tanh"]
    print("\nTanh probabilities:", baseline["probabilities"])
    print("Predictions:", baseline["predictions"])
    print("Early first-layer gradient:", baseline["early_gradient"])
    print("Final first-layer gradient:", baseline["final_gradient"])
    print("\nZero initialization: loss", zero["final_loss"])
    print("Probabilities:", zero["probabilities"])
    for snapshot in zero["symmetry_snapshots"]:
        print("Before update", snapshot["before_update"], ":", snapshot["weights"])
    print("\nThree-class probabilities:")
    for inputs, probabilities in zip(X.tolist(), three["probabilities"]):
        print(inputs, "->", probabilities)
    print("Predictions:", three["predictions"])
    print("Initial loss:", three["initial_loss"], "Final loss:", three["final_loss"])
    print("Probability sums:", three["probability_sums"])
    print("Softmax shift difference:", three["softmax_shift_max_difference"])
    print("\nFull results saved to results.json")


if __name__ == "__main__":
    main()
