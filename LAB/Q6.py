# Q6.py
# Multi-Armed Bandit: Epsilon-Greedy, UCB and Thompson Sampling

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

true_ctr = [0.10, 0.20, 0.15, 0.30, 0.25]
k = len(true_ctr)
rounds = 1000

def reward(arm):
    return 1 if np.random.rand() < true_ctr[arm] else 0

# ---------- Epsilon Greedy ----------
eps = 0.1
Q = np.zeros(k)
N = np.zeros(k)
eps_rewards = []

for t in range(rounds):
    arm = np.random.randint(k) if np.random.rand() < eps else np.argmax(Q)
    r = reward(arm)
    N[arm] += 1
    Q[arm] += (r - Q[arm]) / N[arm]
    eps_rewards.append(r)

# ---------- UCB ----------
Q = np.zeros(k)
N = np.zeros(k)
ucb_rewards = []

for t in range(rounds):
    if t < k:
        arm = t
    else:
        ucb = Q + np.sqrt(2 * np.log(t + 1) / (N + 1e-9))
        arm = np.argmax(ucb)
    r = reward(arm)
    N[arm] += 1
    Q[arm] += (r - Q[arm]) / N[arm]
    ucb_rewards.append(r)

# ---------- Thompson Sampling ----------
success = np.ones(k)
failure = np.ones(k)
ts_rewards = []

for _ in range(rounds):
    samples = np.random.beta(success, failure)
    arm = np.argmax(samples)
    r = reward(arm)
    if r:
        success[arm] += 1
    else:
        failure[arm] += 1
    ts_rewards.append(r)

print("Average CTR")
print("Epsilon-Greedy :", np.mean(eps_rewards))
print("UCB            :", np.mean(ucb_rewards))
print("Thompson       :", np.mean(ts_rewards))

plt.plot(np.cumsum(eps_rewards)/np.arange(1,rounds+1), label="Epsilon")
plt.plot(np.cumsum(ucb_rewards)/np.arange(1,rounds+1), label="UCB")
plt.plot(np.cumsum(ts_rewards)/np.arange(1,rounds+1), label="Thompson")
plt.xlabel("Rounds")
plt.ylabel("Average CTR")
plt.legend()
plt.show()
