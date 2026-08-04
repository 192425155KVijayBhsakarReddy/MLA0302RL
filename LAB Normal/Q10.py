# Q10.py
# Basic Policy Gradient (REINFORCE) - Investment Strategy

import numpy as np

np.random.seed(0)

returns = np.array([0.02, -0.01, 0.05])   # Buy, Hold, Sell
actions = ["Buy", "Hold", "Sell"]

theta = np.zeros(3)      # policy parameters
alpha = 0.1
episodes = 500

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

for _ in range(episodes):
    probs = softmax(theta)
    a = np.random.choice(3, p=probs)
    reward = returns[a] + np.random.normal(0, 0.01)

    grad = -probs
    grad[a] += 1          # gradient of log policy
    theta += alpha * reward * grad

final_probs = softmax(theta)

print("Learned Investment Policy\n")
for act, p in zip(actions, final_probs):
    print(f"{act:5}: {p:.3f}")

best = actions[np.argmax(final_probs)]
print("\nRecommended Action:", best)
