# Policy Evaluation for Warehouse Robot

ROWS = 4
COLS = 4

# Warehouse Grid
# S = Start
# I = Item (+2)
# G = Goal (+5)
# X = Obstacle (-2)
# . = Empty (0)

grid = [
    ['S', '.', 'I', '.'],
    ['.', 'X', '.', '.'],
    ['I', '.', '.', 'G'],
    ['.', '.', 'X', '.']
]

gamma = 0.9        # Discount Factor
theta = 0.01       # Stopping Condition

# Initial Value Function
V = [[0 for j in range(COLS)] for i in range(ROWS)]

# Fixed Policy: Always Move Right, else Move Down
def get_next_state(i, j):

    if j + 1 < COLS:
        return i, j + 1
    elif i + 1 < ROWS:
        return i + 1, j
    else:
        return i, j

# Reward Function
def reward(i, j):

    if grid[i][j] == 'I':
        return 2
    elif grid[i][j] == 'G':
        return 5
    elif grid[i][j] == 'X':
        return -2
    else:
        return 0

# Policy Evaluation
while True:

    delta = 0

    for i in range(ROWS):
        for j in range(COLS):

            ni, nj = get_next_state(i, j)

            r = reward(ni, nj)

            new_value = r + gamma * V[ni][nj]

            delta = max(delta, abs(new_value - V[i][j]))

            V[i][j] = round(new_value, 2)

    if delta < theta:
        break

# Print Warehouse
print("\nWarehouse\n")

for row in grid:
    print(row)

# Print Value Function
print("\nValue Function\n")

for row in V:
    for value in row:
        print(f"{value:6.2f}", end=" ")
    print()
