import numpy as np
import random
import matplotlib.pyplot as plt

arms=5
trials=500
prob=[0.2,0.5,0.7,0.4,0.9]
epsilon=0.1

random_rewards=[]
greedy_rewards=[]
random_total=0
greedy_total=0
Q=np.zeros(arms)
N=np.zeros(arms)

for _ in range(trials):
    arm=random.randint(0,arms-1)
    reward=1 if random.random()<prob[arm] else 0
    random_total+=reward
    random_rewards.append(random_total)

for _ in range(trials):
    if random.random()<epsilon:
        arm=random.randint(0,arms-1)
    else:
        arm=np.argmax(Q)
    reward=1 if random.random()<prob[arm] else 0
    greedy_total+=reward
    greedy_rewards.append(greedy_total)
    N[arm]+=1
    Q[arm]+=((reward-Q[arm])/N[arm])

print("Random Strategy Reward:",random_total)
print("Epsilon-Greedy Reward:",greedy_total)

plt.plot(random_rewards,label="Random")
plt.plot(greedy_rewards,label="Epsilon-Greedy")
plt.title("Random vs Epsilon-Greedy")
plt.xlabel("Trials")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)
plt.show()
