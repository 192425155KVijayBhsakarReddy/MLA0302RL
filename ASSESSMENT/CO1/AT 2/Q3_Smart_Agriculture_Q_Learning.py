import numpy as np
import random
import matplotlib.pyplot as plt

states=[("Dry","Sunny"),("Dry","Rainy"),("Normal","Sunny"),("Normal","Rainy"),("Wet","Sunny"),("Wet","Rainy")]
actions=["Do Nothing","Irrigate","Add Fertilizer"]

Q=np.zeros((len(states),len(actions)))

alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000

history=[]

def reward_fn(state,action):
    soil,weather=states[state]
    r=0
    if soil=="Dry" and action==1:
        r+=10
    elif soil=="Normal" and action==0:
        r+=10
    elif soil=="Wet" and action==0:
        r+=8
    else:
        r-=5
    if weather=="Rainy" and action==1:
        r-=5
    if action==2 and soil!="Wet":
        r+=3
    return r

for ep in range(episodes):
    s=random.randint(0,len(states)-1)
    total=0
    for _ in range(10):
        if random.random()<epsilon:
            a=random.randint(0,len(actions)-1)
        else:
            a=np.argmax(Q[s])
        r=reward_fn(s,a)
        ns=random.randint(0,len(states)-1)
        Q[s,a]+=alpha*(r+gamma*np.max(Q[ns])-Q[s,a])
        s=ns
        total+=r
    history.append(total)

print("Training Completed\n")
print(Q)
print("\nBest Actions\n")
for i in range(len(states)):
    print(states[i],"->",actions[np.argmax(Q[i])])

plt.plot(history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning Smart Agriculture")
plt.grid()
plt.show()
