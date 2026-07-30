import numpy as np
grid_size=5
goal=(4,4)
obstacles=[(1,2),(2,2),(3,1)]
gamma=0.9
states=[(i,j) for i in range(grid_size) for j in range(grid_size)]
actions=[(-1,0),(1,0),(0,-1),(0,1)]
V=np.zeros((grid_size,grid_size))
policy=np.full((grid_size,grid_size)," ")
def move(state,action):
    r,c=state
    nr=max(0,min(grid_size-1,r+action[0]))
    nc=max(0,min(grid_size-1,c+action[1]))
    if (nr,nc) in obstacles:
        return state
    return(nr,nc)
def reward(state):
    if state==goal:
        return 100
    if state in obstacles:
        return -100
    return -1
for _ in range(100):
    newV=V.copy()
    for s in states:
        if s==goal:
            continue
        values=[]
        for a in actions:
            ns=move(s,a)
            values.append(reward(ns)+gamma*V[ns])
        newV[s]=max(values)
    V=newV
symbols=["↑","↓","←","→"]
for s in states:
    if s==goal:
        policy[s]="G"
    elif s in obstacles:
        policy[s]="X"
    else:
        values=[]
        for a in actions:
            ns=move(s,a)
            values.append(reward(ns)+gamma*V[ns])
        policy[s]=symbols[np.argmax(values)]
print("Optimal Policy:\n")
for i in range(grid_size):
    for j in range(grid_size):
        print(policy[i,j],end=" ")
    print()
print("\nState Values:\n")
print(np.round(V,2))
