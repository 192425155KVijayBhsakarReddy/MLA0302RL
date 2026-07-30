import numpy as np
import random

grid=5
start=(0,0)
goal=(4,4)
battery_limit=20
no_fly=[(1,2),(2,2),(3,1)]
actions=[(-1,0),(1,0),(0,-1),(0,1)]
Q=np.zeros((grid,grid,4))
alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000

def move(state,action):
    r,c=state
    nr=max(0,min(grid-1,r+action[0]))
    nc=max(0,min(grid-1,c+action[1]))
    return(nr,nc)

for _ in range(episodes):
    state=start
    battery=battery_limit
    while state!=goal and battery>0:
        r,c=state
        if random.random()<epsilon:
            a=random.randint(0,3)
        else:
            a=np.argmax(Q[r,c])
        ns=move(state,actions[a])
        if ns in no_fly:
            reward=-100
            ns=state
        elif ns==goal:
            reward=100
        else:
            reward=-1
        battery-=1
        if battery==0 and ns!=goal:
            reward-=50
        nr,nc=ns
        Q[r,c,a]+=alpha*(reward+gamma*np.max(Q[nr,nc])-Q[r,c,a])
        state=ns

state=start
battery=battery_limit
path=[state]
visited=set()

while state!=goal and battery>0:
    if state in visited:
        break
    visited.add(state)
    r,c=state
    a=np.argmax(Q[r,c])
    state=move(state,actions[a])
    if state in no_fly:
        break
    path.append(state)
    battery-=1

print("Optimal Path:")
print(path)
print("Battery Remaining:",battery)
print("Goal Reached:",state==goal)
