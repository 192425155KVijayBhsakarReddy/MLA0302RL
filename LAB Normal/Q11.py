# Q11.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

prices = np.array([100,102,101,104,107,105,108,110,109,112],dtype=np.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,32),
            nn.ReLU(),
            nn.Linear(32,3)
        )
    def forward(self,x):
        return self.fc(x)

online = Net()
target = Net()
target.load_state_dict(online.state_dict())

opt = optim.Adam(online.parameters(),lr=0.01)
loss_fn = nn.MSELoss()

memory=[]
gamma=0.95
eps=0.2

for ep in range(100):
    holding=0
    buy_price=0
    for t in range(len(prices)-1):
        state=np.array([prices[t],holding],dtype=np.float32)

        if random.random()<eps:
            action=random.randint(0,2)   #0 Buy 1 Sell 2 Hold
        else:
            with torch.no_grad():
                action=torch.argmax(online(torch.tensor(state))).item()

        reward=0
        if action==0 and holding==0:
            holding=1
            buy_price=prices[t]
        elif action==1 and holding==1:
            reward=prices[t]-buy_price
            holding=0

        next_state=np.array([prices[t+1],holding],dtype=np.float32)
        done=(t==len(prices)-2)
        memory.append((state,action,reward,next_state,done))

        if len(memory)>=16:
            batch=random.sample(memory,16)
            s,a,r,ns,d=random.choice(batch)

            s=torch.tensor(s)
            ns=torch.tensor(ns)

            q=online(s)
            target_q=q.clone().detach()

            with torch.no_grad():
                best=torch.argmax(online(ns)).item()
                next_value=target(ns)[best]

            target_q[a]=r if d else r+gamma*next_value

            loss=loss_fn(q,target_q)
            opt.zero_grad()
            loss.backward()
            opt.step()

    if ep%20==0:
        target.load_state_dict(online.state_dict())

print("Training Completed")

actions=["BUY","SELL","HOLD"]
holding=0
print("\nLearned Actions")
for p in prices[:-1]:
    s=torch.tensor([p,holding],dtype=torch.float32)
    a=torch.argmax(online(s)).item()
    print(f"Price {p:.0f} -> {actions[a]}")
