# Q8.py
# Autonomous Car Navigation using Simple Policies

import random

GRID = 5
START = (0, 0)
GOAL = (4, 4)
OBSTACLES = {(2, 2), (1, 3)}

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

def move(state, action):
    r, c = state
    dr, dc = ACTIONS[action]
    nr = max(0, min(GRID - 1, r + dr))
    nc = max(0, min(GRID - 1, c + dc))
    if (nr, nc) in OBSTACLES:
        return state, -5
    if (nr, nc) == GOAL:
        return (nr, nc), 10
    return (nr, nc), -1

def random_policy(_):
    return random.choice(list(ACTIONS.keys()))

def greedy_policy(state):
    r, c = state
    if r < GOAL[0]:
        return "DOWN"
    if c < GOAL[1]:
        return "RIGHT"
    return "RIGHT"

def simulate(policy, name):
    state = START
    total = 0
    for step in range(30):
        action = policy(state)
        state, reward = move(state, action)
        total += reward
        if state == GOAL:
            print(f"{name}: Reached goal in {step+1} steps | Reward = {total}")
            return
    print(f"{name}: Goal not reached | Reward = {total}")

simulate(random_policy, "Random Policy")
simulate(greedy_policy, "Greedy Policy")
