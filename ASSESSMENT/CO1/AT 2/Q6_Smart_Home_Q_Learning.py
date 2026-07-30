import numpy as np
import random
import matplotlib.pyplot as plt

states=[(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
actions=["Nothing","Heater","AC","Lights OFF","Lights ON"]

Q=np.zeros((len(states),len(actions)))

alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000

rewards=[]

def get_reward(state,action):
    temp,occ=states[state]
    reward=0
    if temp==1:
        reward+=10
    elif temp==0 and action==1:
        reward+=10
    elif temp==2 and action==2:
        reward+=10
    else:
        reward-=5
    if occ==0 and action==3:
        reward+=5
    if occ==1 and action==4:
        reward+=5
    return reward

for episode in range(episodes):
    state=random.randint(0,len(states)-1)
    total=0
    for step in range(10):
        if random.random()<epsilon:
            action=random.randint(0,len(actions)-1)
        else:
            action=np.argmax(Q[state])
        reward=get_reward(state,action)
        next_state=random.randint(0,len(states)-1)
        Q[state,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action])
        state=next_state
        total+=reward
    rewards.append(total)

print("Training Completed\\n")
print(Q)
print("\\nBest Actions\\n")
for i in range(len(states)):
    print(states[i],"->",actions[np.argmax(Q[i])])

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning Smart Home")
plt.grid()
plt.show()
