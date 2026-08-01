# Q14.py
# Dynamic Programming - Policy Iteration (Grid World)

import numpy as np

GRID = 5
GOAL = (4, 4)
gamma = 0.9

actions = [(-1,0),(1,0),(0,-1),(0,1)]
policy = np.zeros((GRID, GRID), dtype=int)
V = np.zeros((GRID, GRID))

def move(r, c, a):
    dr, dc = actions[a]
    nr = max(0, min(GRID-1, r+dr))
    nc = max(0, min(GRID-1, c+dc))
    return nr, nc

stable = False
while not stable:

    # Policy Evaluation
    for _ in range(50):
        newV = V.copy()
        for r in range(GRID):
            for c in range(GRID):
                if (r, c) == GOAL:
                    continue
                nr, nc = move(r, c, policy[r, c])
                reward = 10 if (nr, nc) == GOAL else -1
                newV[r, c] = reward + gamma * V[nr, nc]
        V = newV

    # Policy Improvement
    stable = True
    for r in range(GRID):
        for c in range(GRID):
            if (r, c) == GOAL:
                continue
            old = policy[r, c]
            values = []
            for a in range(4):
                nr, nc = move(r, c, a)
                reward = 10 if (nr, nc) == GOAL else -1
                values.append(reward + gamma * V[nr, nc])
            policy[r, c] = np.argmax(values)
            if old != policy[r, c]:
                stable = False

print("Optimal State Values\n")
print(np.round(V, 2))

print("\nOptimal Policy\n")
arrow = ["↑", "↓", "←", "→"]
for i in range(GRID):
    for j in range(GRID):
        if (i, j) == GOAL:
            print("G", end=" ")
        else:
            print(arrow[policy[i, j]], end=" ")
    print()
