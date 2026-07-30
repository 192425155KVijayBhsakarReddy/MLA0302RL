import numpy as np
import random
import matplotlib.pyplot as plt

SIZE=6
PICKUP=(1,1)
DROP=(5,5)
OBSTACLES=[(2,2),(3,2),(4,4)]

actions=["Up","Down","Left","Right"]
moves=[(-1,0),(1,0),(0,-1),(0,1)]

Q=np.zeros((SIZE,SIZE,2,len(actions)))

alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000
history=[]

def step(state,carry,action):
    x,y=state
    dx,dy=moves[action]
    nx=max(0,min(SIZE-1,x+dx))
    ny=max(0,min(SIZE-1,y+dy))
    if (nx,ny) in OBSTACLES:
        return state,carry,-20
    reward=-1
    if (nx,ny)==PICKUP and carry==0:
        carry=1
        reward=30
    if (nx,ny)==DROP and carry==1:
        reward=100
        return (nx,ny),carry,reward
    return (nx,ny),carry,reward

for ep in range(episodes):
    state=(0,0)
    carry=0
    total=0
    for _ in range(100):
        x,y=state
        if random.random()<epsilon:
            a=random.randint(0,3)
        else:
            a=np.argmax(Q[x,y,carry])
        ns,ncarry,r=step(state,carry,a)
        nx,ny=ns
        Q[x,y,carry,a]+=alpha*(r+gamma*np.max(Q[nx,ny,ncarry])-Q[x,y,carry,a])
        state,carry=ns,ncarry
        total+=r
        if state==DROP and carry==1:
            break
    history.append(total)

print("Training Completed")
print("\nLearned Policy")
for i in range(SIZE):
    row=[]
    for j in range(SIZE):
        if (i,j)==PICKUP:
            row.append("P")
        elif (i,j)==DROP:
            row.append("D")
        elif (i,j) in OBSTACLES:
            row.append("X")
        else:
            row.append(actions[np.argmax(Q[i,j,0])][0])
    print(row)

plt.plot(history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Warehouse Robot Q-Learning")
plt.grid()
plt.show()
