import numpy as np
import random
import matplotlib.pyplot as plt

users=["Action","Comedy","Drama"]
movies=["Action","Comedy","Drama","SciFi","Horror"]

Q=np.zeros((len(users),len(movies)))

alpha=0.1
gamma=0.9
epsilon=0.2
episodes=1000
history=[]

preferences={
0:[0,3],
1:[1],
2:[2]
}

def reward(state,action):
    if action in preferences[state]:
        return 10
    return -5

for ep in range(episodes):
    state=random.randint(0,len(users)-1)
    total=0
    for _ in range(10):
        if random.random()<epsilon:
            action=random.randint(0,len(movies)-1)
        else:
            action=np.argmax(Q[state])
        r=reward(state,action)
        next_state=random.randint(0,len(users)-1)
        Q[state,action]+=alpha*(r+gamma*np.max(Q[next_state])-Q[state,action])
        state=next_state
        total+=r
    history.append(total)

print("Training Completed\n")
print("Q-Table:\n")
print(Q)
print("\nBest Recommendation\n")
for i,u in enumerate(users):
    print(u,"User ->",movies[np.argmax(Q[i])])

plt.plot(history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Movie Recommendation using Q-Learning")
plt.grid()
plt.show()
