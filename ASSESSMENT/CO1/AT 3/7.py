import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam

states=5
actions=3
episodes=500
alpha=0.1
gamma=0.9
epsilon=0.2
bandwidth_limit=70
Q=np.zeros((states,actions))

model=Sequential([
Input(shape=(states,)),
Dense(24,activation="relu"),
Dense(24,activation="relu"),
Dense(actions,activation="linear")
])
model.compile(optimizer=Adam(learning_rate=0.001),loss="mse")

for _ in range(episodes):
    state=random.randint(0,states-1)
    bandwidth=random.randint(30,100)
    if random.random()<epsilon:
        action=random.randint(0,actions-1)
    else:
        action=np.argmax(Q[state])
    reward=30 if bandwidth<=bandwidth_limit else -20
    next_state=random.randint(0,states-1)
    Q[state,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action])
    x=np.zeros((1,states))
    x[0,state]=1
    target=model.predict(x,verbose=0)
    target[0,action]=Q[state,action]
    model.fit(x,target,epochs=1,verbose=0)

print("Q-Table")
print(np.round(Q,2))

print("\nOptimal Routing Policy")
for i in range(states):
    print(f"State {i} -> Route {np.argmax(Q[i])+1}")
