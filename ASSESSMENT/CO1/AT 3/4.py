import numpy as np

grid_size=5
gamma=0.9
theta=0.001
goal=(4,4)
restricted=[(1,2),(2,2),(3,1)]
actions=[(-1,0),(1,0),(0,-1),(0,1)]
symbols=["↑","↓","←","→"]
V=np.zeros((grid_size,grid_size))
policy=np.full((grid_size,grid_size)," ")

def move(state,action):
    r,c=state
    nr=max(0,min(grid_size-1,r+action[0]))
    nc=max(0,min(grid_size-1,c+action[1]))
    if(nr,nc) in restricted:
        return state
    return(nr,nc)

def reward(state):
    if state==goal:
        return 100
    return -1

while True:
    delta=0
    newV=V.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if(i,j)==goal or (i,j) in restricted:
                continue
            values=[]
            for a in actions:
                ns=move((i,j),a)
                values.append(reward(ns)+gamma*V[ns])
            newV[i,j]=max(values)
            delta=max(delta,abs(newV[i,j]-V[i,j]))
    V=newV
    if delta<theta:
        break

for i in range(grid_size):
    for j in range(grid_size):
        if(i,j)==goal:
            policy[i,j]="G"
        elif(i,j) in restricted:
            policy[i,j]="X"
        else:
            values=[]
            for k,a in enumerate(actions):
                ns=move((i,j),a)
                values.append(reward(ns)+gamma*V[ns])
            policy[i,j]=symbols[np.argmax(values)]

print("Optimal Policy:\n")
for row in policy:
    print(" ".join(row))

print("\nState Values:\n")
print(np.round(V,2))
