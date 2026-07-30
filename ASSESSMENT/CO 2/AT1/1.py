import random
states = 5
actions = ["Left", "Right"]
Q = [[0, 0] for _ in range(states)]
alpha = 0.1
gamma = 0.9
epsilon = 0.2
for episode in range(500):
    state = 0
    while state != 4:
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = Q[state].index(max(Q[state]))
        if action == 0:
            next_state = max(0, state - 1)
        else:
            next_state = min(4, state + 1)

        if next_state == 4:
            reward = 100
        else:
            reward = -1

        Q[state][action] = Q[state][action] + alpha * (
            reward + gamma * max(Q[next_state]) - Q[state][action]
        )

        state = next_state
print("Q Table")
for i in range(states):
    print("State", i, ":", Q[i])
print("\nOptimal Route")
state = 0
while state != 4:
    print("State", state, "->", end=" ")
    action = Q[state].index(max(Q[state]))

    if action == 0:
        state = max(0, state - 1)
    else:
        state = min(4, state + 1)

print("Destination Reached (State 4)")
