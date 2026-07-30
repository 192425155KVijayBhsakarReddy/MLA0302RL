import random

states = 2
actions = ["Approve", "Block"]

Q = [[0, 0] for _ in range(states)]

alpha = 0.1
gamma = 0.9
epsilon = 0.2

for episode in range(500):

    state = random.randint(0, 1)

    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        action = Q[state].index(max(Q[state]))

    if state == 0:
        reward = 10 if action == 0 else -10
    else:
        reward = 10 if action == 1 else -10

    Q[state][action] = Q[state][action] + alpha * (
        reward + gamma * max(Q[state]) - Q[state][action]
    )

print("Q Table")
for i in range(states):
    print("State", i, ":", Q[i])

print("\nFraud Detection Policy")

for i in range(states):
    print("State", i, "->", actions[Q[i].index(max(Q[i]))])
