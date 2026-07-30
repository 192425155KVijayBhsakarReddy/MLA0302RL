import numpy as np
import random

states=3
actions=2
Q=np.zeros((states,actions))
alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000

for _ in range(episodes):
    traffic=random.randint(0,2)
    emergency=random.choice([0,0,0,1])
    if emergency:
        action=1
        reward=100
    else:
        if random.random()<epsilon:
            action=random.randint(0,1)
        else:
            action=np.argmax(Q[traffic])
        if traffic==0:
            reward=10 if action==0 else 5
        elif traffic==1:
            reward=15 if action==1 else 8
        else:
            reward=20 if action==1 else 2
    next_state=random.randint(0,2)
    Q[traffic,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[traffic,action])

print("Q-Table:")
print(np.round(Q,2))

signals=["Green","Red"]

print("\nOptimal Policy:")
for i in range(states):
    print(f"Traffic State {i}: {signals[np.argmax(Q[i])]}")

print("\nEmergency Vehicle: Green Signal Immediately")
