import random

states=2
actions=["Increase","Maintain"]
Q=[[0,0] for _ in range(states)]
a,g,e=0.1,0.9,0.2

for _ in range(500):
    s=random.randint(0,1)
    act=random.randint(0,1) if random.random()<e else Q[s].index(max(Q[s]))
    r=10 if (s==1 and act==0) or (s==0 and act==1) else -5
    Q[s][act]+=a*(r+g*max(Q[s])-Q[s][act])

print("Q Table:",Q)
for i in range(states):
    print("State",i,"->",actions[Q[i].index(max(Q[i]))])
