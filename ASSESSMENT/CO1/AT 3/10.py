import numpy as np
import random

states=5
actions=3
episodes=1000
threshold=50
alpha=0.1
gamma=0.9
epsilon=0.2
Q=np.zeros((states,actions))

for _ in range(episodes):
    state=random.randint(0,states-1)
    consumption=random.randint(20,80)
    if random.random()<epsilon:
        action=random.randint(0,actions-1)
    else:
        action=np.argmax(Q[state])
    if action==0:
        consumption-=10
    elif action==1:
        consumption+=5
    reward=20 if consumption<=threshold else -20
    reward+=max(0,threshold-consumption)//5
    next_state=random.randint(0,states-1)
    Q[state,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action])

print("Q-Table:")
print(np.round(Q,2))

print("\nOptimal Policy:")
policies=["Reduce Usage","Maintain Usage","Increase Usage"]
for i in range(states):
    print(f"State {i}: {policies[np.argmax(Q[i])]}")
