"""Plan and follow a route from S to G using A*."""
from heapq import heappop, heappush
from itertools import count

WAREHOUSE_MAP = """#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""


class GoalBasedAgent:
    def __init__(self, ascii_map):
        self.grid = ascii_map.strip().splitlines()
        starts = []
        goals = []
        for r, row in enumerate(self.grid):
            for c, symbol in enumerate(row):
                if symbol == "S":
                    starts.append((r, c))
                elif symbol == "G":
                    goals.append((r, c))
                elif symbol not in ".#":
                    raise ValueError("Unknown map symbol")
        if len(starts) != 1 or len(goals) != 1:
            raise ValueError("Exactly one S and G are required")
        self.start, self.goal = starts[0], goals[0]
        self.position = self.start

    def is_free(self, state):
        r, c = state
        return (0 <= r < len(self.grid) and 0 <= c < len(self.grid[r])
                and self.grid[r][c] != "#")

    def valid_actions(self, state):
        r, c = state
        for action, dr, dc in [("Up", -1, 0), ("Down", 1, 0),
                               ("Left", 0, -1), ("Right", 0, 1)]:
            successor = (r + dr, c + dc)
            if self.is_free(successor):
                yield action, successor

    def heuristic(self, state):
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def formulate_plan(self):
        serial = count()
        frontier = [(self.heuristic(self.position), next(serial), 0, self.position)]
        g = {self.position: 0}
        parent = {self.position: None}
        while frontier:
            _, _, old_g, state = heappop(frontier)
            if old_g != g[state]:
                continue
            if state == self.goal:
                path = [state]
                while parent[state] is not None:
                    state = parent[state]
                    path.append(state)
                return path[::-1]
            for _, successor in self.valid_actions(state):
                new_g = old_g + 1
                if new_g < g.get(successor, float("inf")):
                    g[successor] = new_g
                    parent[successor] = state
                    priority = new_g + self.heuristic(successor)
                    heappush(frontier, (priority, next(serial), new_g, successor))
        return None

    def execute(self, path):
        """Follow the path, checking that each move is allowed."""
        if not path or path[0] != self.position:
            raise ValueError("Plan must start at the current position")
        actions = []
        for next_state in path[1:]:
            candidates = {state: name for name, state in self.valid_actions(self.position)}
            if next_state not in candidates:
                raise ValueError("Invalid step in plan")
            actions.append(candidates[next_state])
            self.position = next_state
        assert self.position == self.goal
        return actions


def main():
    agent = GoalBasedAgent(WAREHOUSE_MAP)
    print("Coordinates: (row,column), zero-based")
    path = agent.formulate_plan()
    if path is None:
        print("No path exists")
    else:
        print("Path:", path)
        print("Length:", len(path) - 1)
        print("Actions:", agent.execute(path))
        print("Reached goal:", agent.position == agent.goal)
    # The agent must also stop when the goal is blocked.
    assert GoalBasedAgent("#####\n#S#G#\n#####").formulate_plan() is None
    print("Blocked-map check: no path exists")


if __name__ == "__main__":
    main()
