import random
import math

prices = [100, 200, 300]
true_reward = [0.4, 0.6, 0.8]

steps = 100

# Epsilon Greedy
epsilon = 0.1
Q = [0, 0, 0]
N = [0, 0, 0]
reward = 0

for t in range(steps):

    if random.random() < epsilon:
        arm = random.randint(0, 2)
    else:
        arm = Q.index(max(Q))

    r = prices[arm] if random.random() < true_reward[arm] else 0

    reward += r
    N[arm] += 1
    Q[arm] += (r - Q[arm]) / N[arm]

print("Epsilon Greedy Revenue =", reward)

# UCB
Q = [0,0,0]
N = [1,1,1]
reward = 0

for t in range(1, steps):

    ucb = []

    for i in range(3):
        value = Q[i] + math.sqrt((2*math.log(t+1))/N[i])
        ucb.append(value)

    arm = ucb.index(max(ucb))

    r = prices[arm] if random.random() < true_reward[arm] else 0

    reward += r
    N[arm] += 1
    Q[arm] += (r-Q[arm])/N[arm]

print("UCB Revenue =", reward)

# Thompson Sampling (Simplified)

success=[1,1,1]
failure=[1,1,1]

reward=0

for t in range(steps):

    samples=[random.betavariate(success[i],failure[i]) for i in range(3)]

    arm=samples.index(max(samples))

    if random.random()<true_reward[arm]:
        reward+=prices[arm]
        success[arm]+=1
    else:
        failure[arm]+=1

print("Thompson Sampling Revenue =",reward)
