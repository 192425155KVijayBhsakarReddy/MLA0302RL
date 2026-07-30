import numpy as np
import random

GRID_SIZE = 5

START = (0, 0)
GOAL = (4, 4)

OBSTACLES = [(1, 2), (2, 2), (3, 1)]

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 500

# Q Table
Q = np.zeros((GRID_SIZE, GRID_SIZE, 4))

def reward(state):

    if state == GOAL:
        return 100

    if state in OBSTACLES:
        return -100

    return -1

def move(state, action):

    r, c = state

    if action == 0:      # UP
        r -= 1

    elif action == 1:    # DOWN
        r += 1

    elif action == 2:    # LEFT
        c -= 1

    elif action == 3:    # RIGHT
        c += 1

    r = max(0, min(GRID_SIZE - 1, r))
    c = max(0, min(GRID_SIZE - 1, c))

    return (r, c)

for ep in range(episodes):

    state = START

    while state != GOAL:

        r, c = state

        # ε-greedy action
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[r, c])

        next_state = move(state, action)

        rw = reward(next_state)

        nr, nc = next_state

        Q[r, c, action] = Q[r, c, action] + alpha * (
            rw +
            gamma * np.max(Q[nr, nc]) -
            Q[r, c, action]
        )

        if next_state in OBSTACLES:
            break

        state = next_state


# -----------------------------
# Display Optimal Path
# -----------------------------
print("\nOptimal Path\n")

state = START

path = [state]

visited = set()

while state != GOAL:

    if state in visited:
        break

    visited.add(state)

    r, c = state

    action = np.argmax(Q[r, c])

    state = move(state, action)

    path.append(state)

print(path)

# -----------------------------
# Display Grid
# -----------------------------
print("\nGrid Representation\n")

for i in range(GRID_SIZE):

    for j in range(GRID_SIZE):

        if (i, j) == START:
            print("S", end=" ")

        elif (i, j) == GOAL:
            print("G", end=" ")

        elif (i, j) in OBSTACLES:
            print("X", end=" ")

        elif (i, j) in path:
            print("*", end=" ")

        else:
            print(".", end=" ")

    print()
