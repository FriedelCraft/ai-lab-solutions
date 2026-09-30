# Lab 3: Goal-Based Agents

## Task 1: Understanding the problem

1. **Environment:** A static warehouse grid. `#` denotes an impassable shelf or boundary; `.`, S, and G are traversable. The model is discrete, deterministic, fully observable, and single-agent. This assumes the complete supplied map is available.
2. **Goal:** Move the vehicle from S to the dispatch location G without crossing obstacles.
3. **Actions:** Up, Down, Left, Right, each changing position by one cell with unit cost. A move is allowed only when its destination is traversable and inside the map.
4. **Information maintained:** Current `(row,column)` position, the map, goal coordinate, frontier, best discovered path costs, predecessor links, and the planned sequence. Position is the changing state; the map and goal are fixed problem data.
5. **Why goal-based:** The agent explicitly represents the destination and searches over possible future states before executing a plan. Its action is selected because it belongs to a goal-reaching route, rather than solely by a fixed rule reacting to the current percept.

**Think About It: doubling the warehouse.** A* remains suitable for a finite known unit-cost grid. If the number of cells doubles, state storage and search work can increase; doubling both dimensions approximately quadruples the cells. With duplicate detection, BFS is O(V+E), and on a grid E is O(V); the number of syntactic paths need not determine runtime. A* commonly takes O(V log V) with a heap on this consistent-heuristic grid. Better heuristics or hierarchical planning may help for much larger maps. Moving obstacles or incomplete observations would introduce additional difficulties, requiring monitoring, replanning, and possibly a richer state representation.

## Task 2: Agent design

| Component | Design |
| --- | --- |
| Environment | The exact ASCII warehouse from the Agents PDF |
| State | Zero-based `(row,column)` vehicle position |
| Goal | The coordinate marked G, `(1,19)` |
| Available actions | Legal neighboring coordinates, with direction names |
| Decision component | A* using Manhattan distance and best-g tracking |
| Execution component | Check and apply each move to the current position |

```text
Warehouse map and observed position + explicit goal
                         |
                         v
             A* decision-making component
                         |
                         v
                Planned action sequence
                         |
                         v
              Check move -> execute move
                         |
                         v
          Updated warehouse position / goal test
```

The map has seven rows, each 21 characters wide. It is a different map from the Search lab and is preserved exactly. A* calculates `g+h`; Manhattan h is a consistent lower bound for four-direction unit-cost motion. The planner retains predecessor links and reconstructs the goal-reaching path. Its `execute` method separately checks every next coordinate against the environment's legal actions before updating position.

## Task 3: Prompt engineering and execution

The implementation specification guiding this AI-assisted solution was:

> Write a documented Python goal-based warehouse agent using the exact supplied map. Represent state as a row/column tuple and provide functions for traversability and legal named actions. Use A* with Manhattan distance, a heap frontier, best path costs, stale-entry checks, and predecessor reconstruction. Print a collision-free path or a clear no-path message. Simulate execution by checking each move, and confirm the final position is G. Keep movement four-directional and unit cost.

The final source is [warehouse_agent.py](warehouse_agent.py).

**Results:** The original map has a valid 20-move path. The agent starts at `(1,1)`, executes the plan, and ends at `(1,19)` with `Reached goal: True`.

```text
Path:
[(1,1),(1,2),(1,3),(1,4),(1,5),(2,5),(2,6),(2,7),
 (1,7),(1,8),(1,9),(1,10),(1,11),(1,12),(1,13),(1,14),
 (1,15),(1,16),(1,17),(1,18),(1,19)]

Actions:
Right, Right, Right, Right, Down, Right, Right, Up,
Right, Right, Right, Right, Right, Right, Right, Right,
Right, Right, Right, Right
```

The horizontal separation is 18 cells. The blocked corridor requires at least one downward and one upward move, giving a lower bound of 20, attained by this path. Every step is checked for legal adjacency and obstacle avoidance. A separate blocked map (`##### / #S#G# / #####`) returns no path.

**1. Did the generated program work on its first execution?** Yes, this program produced the path and passed simulated execution on its first run. That statement refers to this preparation session's program, not a separate student experiment.

**2. How could the specification be improved if a program failed?** Include the exact map, coordinate convention, bounds checks, expected failure behavior, and a request to verify every returned move. Diagnose the specific failing test before modifying the implementation.

**3. Which search algorithm was used?** A* with Manhattan distance. It was specified explicitly, rather than independently selected by the LLM.

**4. Why is it appropriate?** It combines accumulated cost with a valid lower bound and yields a shortest route for this finite graph when best-g updates are handled correctly. It may explore fewer states than uninformed search on some maps; that is not guaranteed on every map.

## Reflection on LLM assistance

Codex translated the design into parsing, frontier management, path reconstruction, and execution checks. Verification mattered because producing coordinates alone would not demonstrate a working agent: the actions must be legal and the final position must satisfy the explicit goal. The reasoning also distinguishes planning over a known static map from operating a physical robot with uncertain sensing or moving obstacles.

The implementation and report were prepared with AI assistance; the student should evaluate the explanations against their own understanding.
