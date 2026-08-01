# Q9.py
# Monte Carlo Policy Evaluation - Call Center

import random

states = ["Low", "Medium", "High"]   # Call load
policy = {"Low": 1, "Medium": 2, "High": 3}  # Representatives assigned

gamma = 0.9
episodes = 1000
returns = {s: [] for s in states}

def simulate(state):
    reps = policy[state]
    if state == "Low":
        reward = 5 - reps
    elif state == "Medium":
        reward = 7 - abs(reps - 2)
    else:
        reward = 10 - abs(reps - 3)
    return reward

for _ in range(episodes):
    state = random.choice(states)
    G = simulate(state)
    returns[state].append(G)

V = {s: sum(returns[s]) / len(returns[s]) for s in states}

print("Estimated State Values (Monte Carlo)\n")
for s in states:
    print(f"{s:6} : {V[s]:.2f}")
