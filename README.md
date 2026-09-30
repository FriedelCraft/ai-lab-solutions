# Artificial Intelligence Laboratory Solutions

Five labs, arranged in the same order and folder style as the reference repository. Each folder contains its question PDF, executable source, and a `submission.md` report with explanations and verified results.

| Lab | Topic | Program | Report |
| --- | --- | --- | --- |
| 1 | Logical planning, including the reference's optional Prolog section | [planner.py](Lab1_Logic/planner.py), [planner.pl](Lab1_Logic/planner.pl) | [submission](Lab1_Logic/submission.md) |
| 2 | A*, BFS, and heuristic investigation | [search_agent.py](Lab2_Search/search_agent.py) | [submission](Lab2_Search/submission.md) |
| 3 | Goal-based warehouse agent | [warehouse_agent.py](Lab3_Agents/warehouse_agent.py) | [submission](Lab3_Agents/submission.md) |
| 4 | XOR, gradients, symmetry, activations, and three-class extension | [neural_xor.py](Lab4_Neural_Models/neural_xor.py) | [submission](Lab4_Neural_Models/submission.md) |
| 5 | First- and second-order Bayesian language models | [language_model.py](Lab5_Bayesian_Networks/language_model.py) | [submission](Lab5_Bayesian_Networks/submission.md) |

## Setup

Use Python 3.10 or later. Only the neural lab needs PyTorch/NumPy; the other Python programs use the standard library.

From this folder, on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy
```

## Run

```powershell
python Lab1_Logic/planner.py
python Lab2_Search/search_agent.py
python Lab3_Agents/warehouse_agent.py
python Lab4_Neural_Models/neural_xor.py
python Lab5_Bayesian_Networks/language_model.py
python -m unittest discover -s Lab5_Bayesian_Networks -v
```

For the optional Prolog checks, install or use [SWI-Prolog](https://www.swi-prolog.org/download/stable), then run:

```powershell
swipl -q -s Lab1_Logic/planner.pl -g run_checks -t halt
```

Programs can also be run from inside their lab folders. Labs 4 and 5 save their results beside their scripts; Lab 5 also saves 20 sampled sentences from each model. The reports include the essential results, so no notebook execution is required to read the submission.

## Verification and conventions

- Logic: all three prescribed tests pass; returned plans are replayed with precondition, state-invariant, and goal checks. The optional queries were executed using the official SWI-Prolog WebAssembly runtime, underlying SWI-Prolog 10.1.15.
- Search: original, adjacent, blocked, and alternative-path tests pass. Returned paths are checked for legal steps and compared with BFS shortest-path lengths. Expansion counts exclude the goal pop and include any reopening.
- Agents: the supplied map is preserved; every planned move is checked during execution. The goal is reached after 20 moves.
- Neural: CPU, PyTorch 2.14.1+cpu, seed 0, Adam, learning rate 0.03, 4,000 full-batch updates. The tanh baseline and three-class extension classify all four inputs correctly. ReLU's 3/4 result is reported as an observed failure, without changing the activation experiment to hide it.
- Bayesian models: seven standard-library unit tests pass, including row normalization, exact counts, boundaries, context dependence, unseen contexts, reproducible sampling, and explicit generation limits.

Coordinates in both grid labs are zero-based `(row, column)`. Path length counts moves, rather than cells. Numerical rounding is for display; full values are saved in `results.json`. Neural results may vary slightly between library versions.

## Scope and assistance

The [reference repository](https://github.com/devam2006/ai-lab-solution) was inspected to establish scope and presentation. These solutions were implemented and tested for this submission; its prose, code, and experimental results were not copied. RAG, transformer, and Ollama exercises are excluded as requested.

These files were prepared with Codex assistance. The quoted implementation specifications in the reports document the requirements guiding generation; they are not claimed to be separate prompts personally sent by the student or a transcript of a separate LLM session. Corrections and test outcomes describe this preparation session. Personal reflections should be reviewed against the student's own understanding before submission.
