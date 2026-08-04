import random

# 5x5 Grid
#  1 = Dirt (+1)
#  0 = Empty
# -1 = Obstacle (-1)

grid = [
    [0, 1, 1, 0, 1],
    [0, -1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, -1, 1, 0],
    [1, 0, 1, 0, 1]
]

ROWS = 5
COLS = 5

actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

# Optimal path
optimal_moves = [
    "RIGHT", "RIGHT", "RIGHT", "RIGHT",
    "DOWN", "LEFT", "DOWN", "DOWN",
    "LEFT", "DOWN", "RIGHT", "RIGHT",
    "RIGHT"
]

# Check valid move
def valid(x, y):
    return 0 <= x < ROWS and 0 <= y < COLS

# Print Grid
def print_grid(rx, ry):
    for i in range(ROWS):
        for j in range(COLS):
            if i == rx and j == ry:
                print("R", end=" ")
            elif grid[i][j] == 1:
                print("D", end=" ")
            elif grid[i][j] == -1:
                print("X", end=" ")
            else:
                print(".", end=" ")
        print()
    print()

# Random Policy
def random_policy(x, y):
    moves = []
    for action, (dx, dy) in actions.items():
        nx = x + dx
        ny = y + dy
        if valid(nx, ny):
            moves.append(action)
    return random.choice(moves)

# Greedy Policy
def greedy_policy(x, y):
    best_action = None
    best_reward = -100

    for action, (dx, dy) in actions.items():
        nx = x + dx
        ny = y + dy

        if valid(nx, ny):
            reward = grid[nx][ny]
            if reward > best_reward:
                best_reward = reward
                best_action = action

    return best_action

# Simulation
def simulate(policy):

    x, y = 0, 0
    reward = 0
    cleaned = set()

    print("\n==============================")
    print(policy.upper(), "POLICY")
    print("==============================")

    print("\nInitial Grid")
    print_grid(x, y)

    if policy == "optimal":
        total_steps = len(optimal_moves)
    else:
        total_steps = 20

    for step in range(total_steps):

        if policy == "random":
            move = random_policy(x, y)

        elif policy == "greedy":
            move = greedy_policy(x, y)

        else:
            move = optimal_moves[step]

        dx, dy = actions[move]
        nx = x + dx
        ny = y + dy

        if valid(nx, ny):
            x = nx
            y = ny

            step_reward = 0

            if grid[x][y] == 1 and (x, y) not in cleaned:
                step_reward = 1
                reward += 1
                cleaned.add((x, y))

            elif grid[x][y] == -1:
                step_reward = -1
                reward -= 1

            print("Step", step + 1)
            print("Move :", move)
            print("Reward :", step_reward)
            print("Total Reward :", reward)
            print_grid(x, y)

    print("Final Reward :", reward)
    print("Dirt Cleaned :", len(cleaned))
    print()

# Run all policies
simulate("random")
simulate("greedy")
simulate("optimal")
