"""Compare A* and BFS on the warehouse map."""
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import hypot

WAREHOUSE_MAP = """#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""


class Grid:
    def __init__(self, ascii_map):
        self.rows = ascii_map.strip().splitlines()
        starts = []
        goals = []
        for r, row in enumerate(self.rows):
            for c, char in enumerate(row):
                if char == "S":
                    starts.append((r, c))
                elif char == "G":
                    goals.append((r, c))
                elif char not in ".#":
                    raise ValueError("Unknown map symbol")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("Map must have exactly one S and one G")
        self.start, self.goal = starts[0], goals[0]

    def free(self, state):
        r, c = state
        return (0 <= r < len(self.rows) and 0 <= c < len(self.rows[r])
                and self.rows[r][c] != "#")

    def neighbors(self, state):
        r, c = state
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            successor = (r + dr, c + dc)
            if self.free(successor):
                yield successor


def manhattan(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean(state, goal):
    return hypot(state[0] - goal[0], state[1] - goal[1])


@dataclass
class Result:
    path: list | None
    expanded: int

    @property
    def length(self):
        if self.path is None:
            return None
        return len(self.path) - 1


def reconstruct(parent, state):
    path = [state]
    while parent[state] is not None:
        state = parent[state]
        path.append(state)
    return path[::-1]


def astar(grid, heuristic=manhattan):
    """Return a path and the number of expansions (excluding the goal)."""
    serial = count()
    frontier = [(heuristic(grid.start, grid.goal), next(serial), 0, grid.start)]
    g = {grid.start: 0}
    parent = {grid.start: None}
    expanded = 0
    while frontier:
        _, _, popped_g, state = heappop(frontier)
        # A better route may have been added after this heap entry.
        if popped_g != g[state]:
            continue
        if state == grid.goal:
            return Result(reconstruct(parent, state), expanded)
        expanded += 1
        for successor in grid.neighbors(state):
            tentative_g = popped_g + 1
            if tentative_g < g.get(successor, float("inf")):
                g[successor] = tentative_g
                parent[successor] = state
                f = tentative_g + heuristic(successor, grid.goal)
                heappush(frontier, (f, next(serial), tentative_g, successor))
    return Result(None, expanded)


def bfs(grid):
    frontier = deque([grid.start])
    parent = {grid.start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state == grid.goal:
            return Result(reconstruct(parent, state), expanded)
        expanded += 1
        for successor in grid.neighbors(state):
            if successor not in parent:
                parent[successor] = state
                frontier.append(successor)
    return Result(None, expanded)


def validate_path(grid, path):
    assert path is not None
    assert path[0] == grid.start and path[-1] == grid.goal
    assert all(grid.free(state) for state in path)
    assert all(manhattan(a, b) == 1 for a, b in zip(path, path[1:]))


def main():
    cases = [
        ("Original", WAREHOUSE_MAP, 40),
        ("Adjacent", "#####\n#SG##\n#####", 1),
        ("Blocked", "#######\n#S....#\n###.###\n#...#G#\n#######", None),
        ("Alternative paths", "#######\n#S....#\n#.....#\n#....G#\n#######", 6),
    ]
    print("Coordinates are (row,column), zero-based. Length counts moves.")
    for name, ascii_map, expected in cases:
        grid = Grid(ascii_map)
        result = astar(grid)
        reference = bfs(grid)
        assert result.length == reference.length == expected
        if result.path is not None:
            validate_path(grid, result.path)
        print(f"\n{name}: found={result.path is not None}")
        print(f"Length: {result.length}, expanded: {result.expanded}")
        print("Path:", result.path)
    print("\nOriginal-map algorithm / heuristic comparison")
    grid = Grid(WAREHOUSE_MAP)
    variants = [
        ("BFS", bfs(grid)),
        ("A* Manhattan", astar(grid)),
        ("A* zero", astar(grid, lambda s, t: 0)),
        ("A* Euclidean", astar(grid, euclidean)),
        ("A* 2*Manhattan", astar(grid, lambda s, t: 2 * manhattan(s, t))),
    ]
    for name, result in variants:
        validate_path(grid, result.path)
        print(f"{name}: found=True, length={result.length}, expanded={result.expanded}")


if __name__ == "__main__":
    main()
