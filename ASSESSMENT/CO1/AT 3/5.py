import numpy as np
import random

grid_size=5
battery_limit=30
deliveries=[(0,4),(2,3),(4,1),(4,4)]
start=(0,0)
Q=np.zeros((grid_size,grid_size,4))
actions=[(-1,0),(1,0),(0,-1),(0,1)]
alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000

def move(state,action):
    r,c=state
    nr=max(0,min(grid_size-1,r+action[0]))
    nc=max(0,min(grid_size-1,c+action[1]))
    return(nr,nc)

for _ in range(episodes):
    state=start
    battery=battery_limit
    completed=[]
    while battery>0:
        r,c=state
        if random.random()<epsilon:
            a=random.randint(0,3)
        else:
            a=np.argmax(Q[r,c])
        ns=move(state,actions[a])
        reward=-1
        if ns in deliveries and ns not in completed:
            reward=20
            completed.append(ns)
        battery-=1
        if battery==0:
            reward-=50
        nr,nc=ns
        Q[r,c,a]+=alpha*(reward+gamma*np.max(Q[nr,nc])-Q[r,c,a])
        state=ns

state=start
battery=battery_limit
completed=[]
path=[state]
while battery>0:
    r,c=state
    a=np.argmax(Q[r,c])
    state=move(state,actions[a])
    path.append(state)
    if state in deliveries and state not in completed:
        completed.append(state)
    battery-=1
    if len(completed)==len(deliveries):
        break

print("Optimal Path:")
print(path)
print("\nDeliveries Completed:",len(completed))
print("Remaining Battery:",battery)
