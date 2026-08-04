# Q15.py
# Monte Carlo Policy Control - Call Center

import numpy as np
import random

states = ["Low", "Medium", "High"]
actions = [1, 2, 3]      # Representatives assigned

Q = {(s, a): 0.0 for s in states for a in actions}
returns = {(s, a): [] for s in states for a in actions}
policy = {s: random.choice(actions) for s in states}

episodes = 1000
epsilon = 0.1

def reward(state, action):
    if state == "Low":
        return 5 - action
    elif state == "Medium":
        return 7 - abs(action - 2)
    else:
        return 10 - abs(action - 3)

for _ in range(episodes):
    state = random.choice(states)

    # ε-greedy action
    if random.random() < epsilon:
        action = random.choice(actions)
    else:
        action = policy[state]

    G = reward(state, action)

    returns[(state, action)].append(G)
    Q[(state, action)] = np.mean(returns[(state, action)])

    # Policy Improvement
    policy[state] = max(actions, key=lambda a: Q[(state, a)])

print("Optimal Policy\n")
for s in states:
    print(f"{s:6} -> Assign {policy[s]} Representative(s)")

print("\nState-Action Values")
for s in states:
    for a in actions:
        print(f"Q({s},{a}) = {Q[(s,a)]:.2f}")
