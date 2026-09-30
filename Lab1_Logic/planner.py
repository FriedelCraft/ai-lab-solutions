"""Find a plan to deliver the package from A to C."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    name: str
    positive_preconditions: frozenset
    negative_preconditions: frozenset
    positive_effects: frozenset
    negative_effects: frozenset

    def applicable(self, state):
        positive_ok = self.positive_preconditions <= state
        negative_ok = self.negative_preconditions.isdisjoint(state)
        return positive_ok and negative_ok

    def apply(self, state):
        if not self.applicable(state):
            raise ValueError(f"Preconditions not satisfied: {self.name}")
        return (state - self.negative_effects) | self.positive_effects


def make_action(name, pre, add, delete, negative_pre=()):
    return Action(name, frozenset(pre), frozenset(negative_pre),
                  frozenset(add), frozenset(delete))


def robot_at(place):
    return f"At(Robot,{place})"


def package_at(place):
    return f"At(Package,{place})"


HOLDING = "Holding(Package)"
INITIAL = frozenset({robot_at("A"), package_at("A")})
GOAL = frozenset({package_at("C")})


def warehouse_actions():
    actions = []
    for src, dst in [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]:
        actions.append(make_action(f"Move({src},{dst})", [robot_at(src)],
                                   [robot_at(dst)], [robot_at(src)]))
    for loc in "ABC":
        actions.append(make_action(f"PickUp(Package,{loc})",
                                   [robot_at(loc), package_at(loc)],
                                   [HOLDING], [package_at(loc)], [HOLDING]))
        actions.append(make_action(f"Drop(Package,{loc})",
                                   [robot_at(loc), HOLDING],
                                   [package_at(loc)], [HOLDING]))
    return actions


def bfs_plan(initial, goal, actions):
    """Return a shortest action sequence, [] if already solved, or None."""
    initial, goal = frozenset(initial), frozenset(goal)
    frontier = deque([initial])
    parent = {initial: None}
    while frontier:
        state = frontier.popleft()
        if goal <= state:
            plan = []
            while parent[state] is not None:
                previous_state, action = parent[state]
                plan.append(action)
                state = previous_state
            return plan[::-1]
        for action in actions:
            if action.applicable(state):
                successor = action.apply(state)
                if successor not in parent:
                    parent[successor] = (state, action)
                    frontier.append(successor)
    return None


def verify_plan(initial, goal, plan):
    """Check each action and keep the states for the printed trace."""
    state = frozenset(initial)
    states = [state]
    for action in plan:
        assert action.applicable(state), action.name
        state = action.apply(state)
        assert sum(robot_at(loc) in state for loc in "ABC") == 1
        assert sum(package_at(loc) in state for loc in "ABC") + (HOLDING in state) == 1
        states.append(state)
    assert frozenset(goal) <= state
    return states


def main():
    actions = warehouse_actions()
    print("Initially applicable:", [a.name for a in actions if a.applicable(INITIAL)])
    # Moving the robot alone should not count as delivering the package.
    extra = make_action("Move(A,B) [irrelevant duplicate]", [robot_at("A")],
                        [robot_at("B")], [robot_at("A")])
    without_pickup = [a for a in actions if not a.name.startswith("PickUp")]
    cases = [
        ("A: original", actions, True),
        ("B: no pickup", without_pickup, False),
        ("C: irrelevant move", actions + [extra], True),
    ]
    for name, available, expected in cases:
        plan = bfs_plan(INITIAL, GOAL, available)
        assert (plan is not None) == expected
        print(f"\nTest {name}\nInitial: {sorted(INITIAL)}\nGoal: {sorted(GOAL)}")
        if plan is None:
            print("No plan found")
            continue
        states = verify_plan(INITIAL, GOAL, plan)
        print("Plan:", [a.name for a in plan])
        print("S0:", sorted(states[0]))
        for i, (action, state) in enumerate(zip(plan, states[1:]), 1):
            print(f"S{i} after {action.name}: {sorted(state)}")
        print("Valid: True")
    moves = {action.name: action for action in actions}
    robot_only = moves["Move(A,B)"].apply(INITIAL)
    robot_only = moves["Move(B,C)"].apply(robot_only)
    assert robot_at("C") in robot_only and not GOAL <= robot_only
    print("\nRobot at C without package: goal is False (checked)")
    assert bfs_plan(GOAL, GOAL, []) == []
    # Check that an absent-fact precondition is enforced too.
    blocked = make_action("negative-precondition check", [], [], [], [HOLDING])
    assert not blocked.applicable(frozenset({HOLDING}))


if __name__ == "__main__":
    main()
