import random

states = 3
actions = ["Movie", "Music","Anime"]

Q = [[0, 0, 0] for _ in range(states)]

alpha = 0.1
gamma = 0.9
epsilon = 0.1

for episode in range(500):

    state = random.randint(0, 2)

    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        action = Q[state].index(max(Q[state]))

    if state == action:
        reward = 10
    else:
        reward = -5

    Q[state][action] = Q[state][action] + alpha * (
        reward + gamma * max(Q[state]) - Q[state][action]
    )

print("Q Table")
for i in range(states):
    print("State", i, ":", Q[i])

print("\nRecommendation Policy")

for i in range(states):
    print("State", i, "-> Recommend", actions[Q[i].index(max(Q[i]))])

