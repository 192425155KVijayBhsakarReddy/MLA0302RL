import numpy as np
import random
import matplotlib.pyplot as plt

SIZE=5
GOAL=(4,4)
OBSTACLES=[(1,1),(2,2),(3,1)]

actions=["Up","Down","Left","Right"]
moves=[(-1,0),(1,0),(0,-1),(0,1)]

Q=np.zeros((SIZE,SIZE,len(actions)))

alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000
history=[]

def step(state,action):
    x,y=state
    dx,dy=moves[action]
    nx=max(0,min(SIZE-1,x+dx))
    ny=max(0,min(SIZE-1,y+dy))
    if (nx,ny) in OBSTACLES:
        return state,-10
    if (nx,ny)==GOAL:
        return (nx,ny),100
    return (nx,ny),-1

for ep in range(episodes):
    state=(0,0)
    total=0
    while state!=GOAL:
        x,y=state
        if random.random()<epsilon:
            a=random.randint(0,3)
        else:
            a=np.argmax(Q[x,y])
        ns,r=step(state,a)
        nx,ny=ns
        Q[x,y,a]+=alpha*(r+gamma*np.max(Q[nx,ny])-Q[x,y,a])
        state=ns
        total+=r
    history.append(total)

print("Training Completed\n")
print("Learned Policy:\n")
for i in range(SIZE):
    row=[]
    for j in range(SIZE):
        if (i,j)==GOAL:
            row.append("G")
        elif (i,j) in OBSTACLES:
            row.append("X")
        else:
            row.append(actions[np.argmax(Q[i,j])][0])
    print(row)

plt.plot(history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning Strategy Game")
plt.grid()
plt.show()
