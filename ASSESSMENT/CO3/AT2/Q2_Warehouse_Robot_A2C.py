"""
QUESTION 2 (10 Marks):
Design and implement an Actor-Critic (A2C/A3C) model for an autonomous warehouse robot that optimizes package collection
and delivery. Compare the performance of Vanilla Policy Gradient and Actor-Critic algorithms based on reward, training stability,
and task completion time.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Categorical

class WarehouseEnv:
    def __init__(self): self.reset()
    def reset(self): self.pos = [5,5]; self.pkg = 4; self.has_pkg = False; self.t = 0; return self._s()
    def _s(self): return np.array([self.pos[0]/5.0, self.pos[1]/5.0, self.pkg/4.0, float(self.has_pkg)], dtype=np.float32)
    def step(self, a):
        r, c = self.pos
        if a==0 and r>0: r-=1
        elif a==1 and r<5: r+=1
        elif a==2 and c>0: c-=1
        elif a==3 and c<5: c+=1
        elif a==4:
            if not self.has_pkg and [r,c] in [[1,1],[1,4],[4,1]]: self.has_pkg = True
            elif self.has_pkg and [r,c] == [0,0]: self.has_pkg = False; self.pkg -= 1
        self.pos = [r,c]; self.t += 1
        r_val = 20.0 if (a==4 and not self.has_pkg and [r,c]==[0,0]) else -0.1
        return self._s(), r_val, self.t >= 100 or self.pkg == 0

class ACNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.pi = nn.Linear(64, 5); self.v = nn.Linear(64, 1)
    def forward(self, x):
        h = self.fc(x); return torch.softmax(self.pi(h), -1), self.v(h)

def train(algo="VPG"):
    env = WarehouseEnv(); net = ACNet(); opt = optim.Adam(net.parameters(), lr=1e-3)
    rewards, steps_list = [], []
    for ep in range(150):
        s = env.reset(); lps, vals, rews = [], [], []
        done = False
        while not done:
            p, v = net(torch.FloatTensor(s))
            d = Categorical(p); a = d.sample()
            s, r, done = env.step(a.item())
            lps.append(d.log_prob(a)); vals.append(v); rews.append(r)
        G, returns = 0, []
        for r in reversed(rews): G = r + 0.99*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        if algo == "VPG":
            loss = torch.stack([-lp * g for lp, g in zip(lps, returns)]).sum()
        else: # A2C / A3C
            vals = torch.cat(vals).squeeze()
            adv = returns - vals.detach()
            loss = (-(torch.stack(lps) * adv).mean() + 0.5*nn.functional.mse_loss(vals, returns))
        opt.zero_grad(); loss.backward(); opt.step()
        rewards.append(sum(rews)); steps_list.append(env.t)
    return rewards, steps_list

r_vpg, s_vpg = train("VPG")
r_a2c, s_a2c = train("A2C")
r_a3c, s_a3c = train("A3C")

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(r_vpg, label="VPG"); plt.plot(r_a2c, label="A2C"); plt.plot(r_a3c, label="A3C"); plt.title("Cumulative Reward"); plt.legend()
plt.subplot(122); plt.plot(s_vpg, label="VPG"); plt.plot(s_a2c, label="A2C"); plt.plot(s_a3c, label="A3C"); plt.title("Completion Time (Steps)")
plt.tight_layout(); plt.show()
print(f"Final Rewards -> VPG: {np.mean(r_vpg[-20:]):.2f}, A2C: {np.mean(r_a2c[-20:]):.2f}, A3C: {np.mean(r_a3c[-20:]):.2f}")
