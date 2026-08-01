# Q13.py
# Q-Learning for a Simple Grid-Based Game (Pac-Man Style)

import numpy as np
import random

GRID = 5
START = (0,0)
FOOD = (4,4)
GHOST = (2,2)

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
    for _ in range(100):
        a = choose(state)
        r,c = state
        dr,dc = actions[a]
        nr = max(0,min(GRID-1,r+dr))
        nc = max(0,min(GRID-1,c+dc))
        next_state = (nr,nc)

        reward = -0.1
        if next_state == FOOD:
            reward = 10
        elif next_state == GHOST:
            reward = -10

        Q[r,c,a] += alpha * (
            reward +
            gamma * np.max(Q[nr,nc]) -
            Q[r,c,a]
        )

        state = next_state
        if state == FOOD or state == GHOST:
            break

print("Learned Policy\n")
arrow = ["↑","↓","←","→"]
for i in range(GRID):
    for j in range(GRID):
        if (i,j) == FOOD:
            print("F", end=" ")
        elif (i,j) == GHOST:
            print("G", end=" ")
        else:
            print(arrow[np.argmax(Q[i,j])], end=" ")
    print()
