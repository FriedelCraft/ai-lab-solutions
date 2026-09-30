# Lab 1: Logical Reasoning for Planning

## Task 0: Problem specification

The locations are A, B, and C. The only direct connections are A-B, B-A, B-C, and C-B. The robot and package begin at A:

`I = {At(Robot,A), At(Package,A)}`

The delivery goal is `G = {At(Package,C)}`. The robot reaching C is insufficient: the package must have been dropped there.

States are sets of propositions. A proposition absent from the state is treated as false (the closed-world assumption). The package is either at exactly one location or held by the robot; the robot occupies exactly one location. The environment is deterministic and does not change independently of the robot.

| Action family | Positive preconditions | Negative preconditions | Add effects | Delete effects |
| --- | --- | --- | --- | --- |
| Move(x,y), for a listed connection | At(Robot,x) | None | At(Robot,y) | At(Robot,x) |
| PickUp(Package,x), x in A,B,C | At(Robot,x), At(Package,x) | Holding(Package) must be absent | Holding(Package) | At(Package,x) |
| Drop(Package,x), x in A,B,C | At(Robot,x), Holding(Package) | None | At(Package,x) | Holding(Package) |

The additional negative pickup precondition explicitly rules out picking up an already held package; it does not alter any reachable valid state. Move does not add a package-location proposition: while holding, the package remains represented by `Holding(Package)`.

The ten grounded actions are Move(A,B), Move(B,A), Move(B,C), Move(C,B), PickUp at A/B/C, and Drop at A/B/C. Initially, only Move(A,B) and PickUp(Package,A) are applicable.

**Applicability question:** PickUp(Package,A) is applicable because both co-location facts hold and Holding is absent. Drop(Package,C) is inapplicable because neither At(Robot,C) nor Holding(Package) holds. Membership in the action list alone never establishes applicability.

## Task 1: Plan constructed from the logical specification

| State | Action just taken | Facts |
| --- | --- | --- |
| S0 | None | At(Robot,A), At(Package,A) |
| S1 | PickUp(Package,A) | At(Robot,A), Holding(Package) |
| S2 | Move(A,B) | At(Robot,B), Holding(Package) |
| S3 | Move(B,C) | At(Robot,C), Holding(Package) |
| S4 | Drop(Package,C) | At(Robot,C), At(Package,C) |

This is a four-action plan. At least one pickup, two connected movements, and one drop are necessary, so it is shortest under unit action costs. Moving to B before pickup would leave the package at A and prevent PickUp(Package,B).

## Task 2: Implementation specification and code

The specification guiding the AI-assisted implementation was:

> Implement the A-B-C warehouse planner in Python using sets of logical propositions. Each action has a name, positive and negative preconditions, add effects, and delete effects. Check all preconditions, then remove delete effects and add add effects. Use BFS with visited immutable states to find a shortest plan. Return an empty plan when the goal already holds and report no plan when the frontier is exhausted. Print the action sequence and every resulting state. Run the original, no-pickup, and irrelevant-action tests; independently replay every returned plan and check the package-delivery goal.

The final program is [planner.py](planner.py). `Action.applicable` implements logical applicability; `Action.apply` checks it and applies `(state - delete_effects) | add_effects`. `bfs_plan` uses a FIFO `deque`, with immutable `frozenset` keys and parent links. The goal test is subset inclusion, `goal <= state`, because extra true propositions do not invalidate delivery.

Visited states stop repeated movement and pickup/drop cycles from creating infinite searches. The state space is finite. With unit action costs, BFS finds a shortest action sequence, though its memory requirements grow on larger problems.

## Task 3: Executed tests

All tests use the initial state and goal specified above.

| Test | Available actions | Plan found? | Returned plan | Verification |
| --- | --- | --- | --- | --- |
| A: original | All ten actions | Yes | PickUp(A), Move(A,B), Move(B,C), Drop(C) | Every precondition and state invariant holds; package at C |
| B: impossible | All pickup actions removed | No | `None`; prints `No plan found` | Finite search ends without inventing a pickup |
| C: irrelevant actions | Original actions plus a duplicate robot-only Move(A,B) | Yes | Same four-action plan | Robot-only movement never substitutes for delivery |

Test C also executes Move(A,B), Move(B,C) without pickup. Its final state is `{At(Robot,C), At(Package,A)}`, so the delivery goal is false. This directly checks the distinction between the robot and package positions.

`verify_plan` replays each returned action, checks its preconditions, and confirms exactly one robot location and exactly one package-location/holding condition after every step. The final goal is checked separately. Additional small checks confirm an already satisfied goal returns `[]` and a negative precondition actually blocks an action.

## Task 4: Logic and search

The missing step in the diagram is **apply the action's effects, if the action is applicable**:

```text
Current state
  -> check positive and negative preconditions
  -> apply the applicable action's delete/add effects
  -> generate successor state
  -> put unseen successors in the BFS frontier
  -> test the goal when a state is selected
```

Logic establishes which transitions are permitted and what propositions each transition changes. Search chooses the order in which permitted alternatives are explored. A correct queue cannot compensate for an incorrect action model, and correct action rules alone do not choose a complete plan.

## Task 5: Checking the plan's explanation

For pickup, both robot and package are at A. For Move(A,B), the robot is at A; after that move it is at B, satisfying Move(B,C). Finally the robot is at C and Holding(Package) holds, satisfying Drop(Package,C). The resulting state contains the goal proposition.

Executed transitions and explicit invariant checks provide stronger evidence than a generated verbal explanation because each premise and effect is actually evaluated. They still depend on the correctness of the encoded rules: execution proves agreement with that model, not automatically with a real warehouse. The explanation is useful for interpretation but should be compared with the replayed states.

## Reflection questions

1. **Why specify preconditions/effects first?** They define the legal problem before implementation and give precise criteria against which generated code can be checked.
2. **Example of missing checks:** Dropping at C without holding the package could create `At(Package,C)` from nothing and falsely satisfy delivery.
3. **Why a reasonable-looking plan can fail:** Its actions may require facts absent at execution time, such as PickUp(Package,B) after the robot moved there alone.
4. **LLM contribution:** Codex generated the action representation, BFS implementation, and test/replay code from the lab constraints.
5. **Independent checks needed:** The action model, every replayed transition, package invariants, impossibility result, and final goal were checked by executable assertions and inspection.
6. **Where is logical reasoning?** In precondition satisfaction, add/delete updates, and the goal entailment check under the stated closed-world model.
7. **How is planning related to search?** States are logical fact sets; applicable actions form edges; BFS searches these edges for a goal-reaching sequence.

The code and explanations were prepared with AI assistance. This report records verification of this implementation rather than inventing a student's prior prompting history.

## Optional extension completed in the reference: Prolog, Tasks 6-8

The final file is [planner.pl](planner.pl). Its facts list the four direct connections; `can_move(X,Y) :- connected(X,Y).` and `valid_move(X,Y) :- connected(X,Y).` express the corresponding rule implications.

The following queries were executed in the official SWI-Prolog WebAssembly build (underlying version 10.1.15):

| Query | Observed answer | Reason |
| --- | --- | --- |
| can_move(a,b) | true | connected(a,b) is a fact |
| can_move(a,c) | false | No direct connected(a,c) fact or transitive rule |
| valid_move(a,b) | true | Listed direct connection |
| valid_move(b,c) | true | Listed direct connection |
| valid_move(a,c) | false | Unsupported direct move |
| reduce_speed | true | wet_road implies slippery, which implies reduce_speed |

For Task 8, the logical reasoning is:

`wet_road AND (wet_road -> slippery) => slippery; slippery AND (slippery -> reduce_speed) => reduce_speed`.

Thus a stated fact plus two rule applications establish the conclusion. The movement predicates verify direct connectivity only; they do not check current robot location, pickup, holding, or the complete delivery goal. Python's full state replay performs those additional checks.

**Prolog reflection:** A fact states an unconditional proposition; a rule derives its head when its body can be established. A query asks whether the program can prove a goal using facts, unification, and rule resolution. A separate Prolog model can reject unsupported candidate movements and reduce dependence on the planner's implementation. Independence helps detect mistakes only when the verifier's rules cover the property being checked; sharing a faulty domain assumption would still permit a false conclusion. Prolog's operational semantics, including backtracking and negation as failure, should not be equated with unrestricted classical logic.
