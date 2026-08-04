# Q12.py
# SARSA - Robot Vacuum Cleaner

import numpy as np
import random

GRID = 5
START = (0, 0)
DIRT = {(1,1), (2,3), (4,4)}
OBST = {(1,3), (3,2)}

actions = [(-1,0),(1,0),(0,-1),(0,1)]
Q = np.zeros((GRID, GRID, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

def choose(state):
    if random.random() < epsilon:
        return random.randint(0,3)
    return np.argmax(Q[state[0], state[1]])

for _ in range(episodes):
    state = START
    cleaned = set()
    action = choose(state)

    for _ in range(100):
        r, c = state
        dr, dc = actions[action]
        nr = max(0, min(GRID-1, r+dr))
        nc = max(0, min(GRID-1, c+dc))
        next_state = (nr, nc)

        reward = -0.1
        if next_state in OBST:
            reward = -1
        elif next_state in DIRT and next_state not in cleaned:
            reward = 1
            cleaned.add(next_state)

        next_action = choose(next_state)

        Q[r,c,action] += alpha * (
            reward +
            gamma * Q[nr,nc,next_action] -
            Q[r,c,action]
        )

        state = next_state
        action = next_action

        if len(cleaned) == len(DIRT):
            break

print("Learned Policy\n")
names = ["↑","↓","←","→"]
for i in range(GRID):
    for j in range(GRID):
        if (i,j) in OBST:
            print("X", end=" ")
        elif (i,j) in DIRT:
            print("D", end=" ")
        else:
            print(names[np.argmax(Q[i,j])], end=" ")
    print()
