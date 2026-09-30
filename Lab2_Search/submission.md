# Lab 2: Search and A*

## Task 0: Search formulation

| Component | Specification |
| --- | --- |
| States S | Traversable `(row,column)` coordinates in the fixed map |
| Actions A | Up, Down, Left, Right |
| Transition T | Add the action's coordinate offset if the target is within bounds and not `#` |
| Initial state s0 | `(1,1)`, marked S |
| Goal set G | `{(7,15)}`, marked G |
| Cost c | One per legal move |

Coordinates are zero-based; rows increase downward. The state needs only position because the static map and goal are fixed problem data. An action is invalid if it leaves the map or enters an obstacle. The problem is deterministic: each legal state/action pair has one successor. A solution is a finite legal path from S to G; an optimal solution has the fewest moves.

## Tasks 1-2: Agent design and implementation specification

The map is a list of strings, and positions are tuples. `Grid.neighbors` checks Up, Down, Left, Right in that order. A min-heap frontier stores `(f, insertion_counter, g, state)`. The counter gives FIFO tie breaking for equal f. A `g` dictionary stores the cheapest discovered cost; `parent` records predecessor links for reconstruction. The goal is checked after removing a valid entry from the frontier.

The AI-assisted implementation was guided by this specification:

> Implement A* for the exact warehouse map using four-direction movement and unit cost. Use Manhattan distance. Store g, calculate f=g+h, use a heap frontier with deterministic tie breaking, discard stale entries, and update predecessor links when a cheaper path is found. Reconstruct the path and print found status, path, move count, and expansion count. Implement BFS on the same grid, run original/adjacent/blocked/alternative-path tests, and compare zero, Euclidean, and twice-Manhattan heuristics. Check each returned path against the map and compare shortest-path lengths with BFS.

The final source is [search_agent.py](search_agent.py). Stale entries are discarded when the g value stored in the heap no longer equals the best recorded g. A cheaper path can reopen a state. This also permits the twice-Manhattan experiment without pretending its heuristic is consistent.

An expansion means generating a selected state's successors. The goal pop is excluded because its successors are not generated. Re-expansions are counted again. Failed searches report length `None`, rather than zero, to distinguish failure from a valid zero-move solution.

## Task 3: Systematic testing results

| Test | Found? | Length in moves | Expanded | Check |
| --- | --- | --- | --- | --- |
| Original warehouse | Yes | 40 | 63 | Legal path; equals BFS shortest length |
| Adjacent goal | Yes | 1 | 1 | Exactly one move |
| Inaccessible goal | No | None | 9 | Frontier exhausted and returned failure |
| Alternative paths | Yes | 6 | 14 | Equals BFS shortest length and Manhattan lower bound |

The adjacent and blocked maps are exactly the PDF examples. The alternative-path map is:

```text
#######
#S....#
#.....#
#....G#
#######
```

The returned paths, with both endpoints included, are:

```text
Original:
[(1,1),(1,2),(1,3),(1,4),(1,5),(2,5),(3,5),(4,5),
 (5,5),(5,6),(5,7),(5,8),(5,9),(5,10),(5,11),(5,12),(5,13),
 (4,13),(3,13),(3,12),(3,11),(3,10),(3,9),(3,8),(3,7),
 (2,7),(1,7),(1,8),(1,9),(1,10),(1,11),(1,12),(1,13),(1,14),
 (1,15),(2,15),(3,15),(4,15),(5,15),(6,15),(7,15)]

Adjacent: [(1,1),(1,2)]
Blocked: None
Alternative: [(1,1),(2,1),(3,1),(3,2),(3,3),(3,4),(3,5)]
```

`validate_path` checks correct endpoints, traversability of every coordinate, and Manhattan distance one between consecutive coordinates. BFS provides an independent algorithm for checking optimal lengths in the prescribed tests.

## Task 4: Inspecting A*

| Concept | Where it appears |
| --- | --- |
| State | Tuples in `Grid.start`, `Grid.goal`, and the frontier |
| Action | The four offsets in `Grid.neighbors` |
| Transition | Construction of `successor` in `Grid.neighbors` |
| Goal test | `state == grid.goal` in `astar` |
| g(n) | `g` dictionary and `tentative_g = popped_g + 1` |
| h(n) | `manhattan`, or the supplied `heuristic` function |
| f(n) | `f = tentative_g + heuristic(successor, grid.goal)` |
| Frontier | `heapq.heappush` / `heapq.heappop` |
| Reached states | Keys of `g`; only cheaper revisits are accepted |
| Path reconstruction | `reconstruct(parent,state)` |

(a) The frontier is a min-heap priority queue. (b) The smallest f is selected; insertion order breaks ties. (c) h is computed when inserting the start and each improved successor. (d) f is explicitly calculated as g+h. (e) Equal/worse g revisits and stale heap entries are discarded, while better g values can reopen states.

## Task 5: A* versus BFS

| Measure | BFS | A* with Manhattan |
| --- | --- | --- |
| Solution found | Yes | Yes |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

Both return shortest paths. Neither expands fewer states on this map under the documented ordering. The obstacles require substantial detours, and Manhattan distance does not encode them. A* can reduce exploration when its estimates distinguish promising routes effectively, but this result supplies no evidence of an expansion advantage here.

## Task 6: Heuristic investigation

Manhattan distance is `abs(r-rG)+abs(c-cG)`. Four-direction movement must correct both coordinate differences one unit at a time, so this is a lower bound on path cost. Obstacles can only lengthen the route. It is also consistent: one legal unit move can change h by at most one.

| Heuristic/algorithm | Found? | Length | Expanded |
| --- | --- | --- | --- |
| BFS | Yes | 40 | 63 |
| Manhattan | Yes | 40 | 63 |
| Zero | Yes | 40 | 63 |
| Euclidean | Yes | 40 | 63 |
| Twice Manhattan | Yes | 40 | 66 |

With h=0, A* becomes uniform-cost search. For this unit-cost graph and FIFO ties it explores in BFS order. Euclidean distance is an admissible, consistent lower bound but no larger than Manhattan, so it provides weaker guidance for four-direction movement. Neither heuristic has an expansion advantage in this run.

Twice Manhattan can overestimate the true cost and is not generally admissible or consistent. It found a 40-move path here but performed 66 expansions, including any reopened states. One successful optimal result does not restore its general optimality guarantee. A weak underestimate preserves correctness but may waste work; an overestimate may favor a route that is not cheapest. Exact counts depend on tie breaking, stale-entry handling, and whether the goal pop is counted.

## Final reflection

1. **Problem formulation:** Specifying state, legal transitions, goal, and cost gives meaning to the code and defines what must be verified. Without that specification, a program could return a plausible path that crosses a shelf or optimizes the wrong quantity.
2. **Informed search:** A* uses a problem-specific estimate of remaining cost as well as accumulated cost. BFS uses only depth; A* uses f=g+h to decide which frontier entry to select.
3. **Why the heuristic matters:** Its admissibility/consistency affect guarantees, while its strength and the map affect exploration. Manhattan is a useful lower bound, not the exact distance through obstacles. The equal BFS/A* counts show that informed search need not win every instance.
4. **LLM contribution:** Codex produced the heap, parent-link reconstruction, comparison routines, and assertions. The implementation was inspected for cheaper-path updates, stale entries, frontier exhaustion, and unit-cost conventions, then executed on known-answer cases.
5. **Risks of untested generated code:** A program can omit a cheaper update, confuse path cells with moves, fail to terminate on unreachable goals, or claim optimality under an overestimating heuristic. The adjacent, blocked, and BFS comparison tests target these mistakes.

This report describes the generated implementation and measured results. It does not claim that the student personally understood every data structure before this preparation session.
