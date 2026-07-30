import numpy as np
import matplotlib.pyplot as plt

episodes=100
rewards=np.random.randint(0,100,episodes)
cumulative=np.cumsum(rewards)

plt.plot(cumulative)
plt.title("Cumulative Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
