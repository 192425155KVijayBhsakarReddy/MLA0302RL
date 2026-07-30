import numpy as np
import random
import matplotlib.pyplot as plt

num_arms = 5
trials = 500

true_rewards = [0.2, 0.5, 0.7, 0.4, 0.9]

epsilons = [0.1, 0.3, 0.9]

results = {}

def epsilon_greedy(epsilon):

    Q = np.zeros(num_arms)       # Estimated rewards
    N = np.zeros(num_arms)       # Number of selections

    rewards = []

    total_reward = 0

    for t in range(trials):

        # Exploration
        if random.random() < epsilon:
            action = random.randint(0, num_arms - 1)

        # Exploitation
        else:
            action = np.argmax(Q)

        # Generate reward
        reward = 1 if random.random() < true_rewards[action] else 0

        total_reward += reward

        rewards.append(total_reward)

        # Update estimates
        N[action] += 1

        Q[action] = Q[action] + (reward - Q[action]) / N[action]

    return rewards, total_reward

# -----------------------------
# Run for Different ε Values
# -----------------------------
for e in epsilons:

    rewards, total = epsilon_greedy(e)

    results[e] = rewards

    print(f"Epsilon = {e}")
    print("Total Reward =", total)
    print()

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(8,5))

for e in epsilons:
    plt.plot(results[e], label=f"ε = {e}")

plt.title("ε-Greedy Multi-Armed Bandit")
plt.xlabel("Trials")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.show()
