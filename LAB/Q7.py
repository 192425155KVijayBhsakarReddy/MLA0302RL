# Q7.py
# Bellman State-Value Function for Delivery Robot

import numpy as np
import matplotlib.pyplot as plt

GRID = 5
GOAL = (4, 4)
gamma = 0.9

V = np.zeros((GRID, GRID))

def next_state(r, c, a):
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    dr, dc = moves[a]
    nr = max(0, min(GRID-1, r+dr))
    nc = max(0, min(GRID-1, c+dc))
    return nr, nc

# Iterative Bellman Evaluation (uniform random policy)
for _ in range(100):
    newV = V.copy()
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == GOAL:
                newV[r, c] = 10
                continue
            value = 0
            for a in range(4):
                nr, nc = next_state(r, c, a)
                reward = 10 if (nr, nc) == GOAL else -1
                value += 0.25 * (reward + gamma * V[nr, nc])
            newV[r, c] = value
    V = newV

print("State Value Function:\n")
print(np.round(V, 2))

plt.imshow(V, cmap="viridis")
plt.colorbar(label="State Value")
plt.title("Bellman State Values")
plt.show()
