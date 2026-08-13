"""
QUESTION 8 (10 Marks):
Design and implement an A3C-based Reinforcement Learning framework for dynamic cloud resource allocation.
Compare the algorithm with Vanilla Policy Gradient using metrics such as response time, CPU utilization, and resource efficiency.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Categorical

class CloudEnv:
    def __init__(self): self.reset()
    def reset(self): self.load = np.random.uniform(0.2, 0.9); self.alloc = 0.5; self.t = 0; return self._s()
    def _s(self): return np.array([self.load, self.alloc], dtype=np.float32)
    def step(self, a):
        if a == 0: self.alloc = max(0.1, self.alloc - 0.2)
        elif a == 2: self.alloc = min(1.0, self.alloc + 0.2)
        self.load = np.clip(self.load + np.random.normal(0, 0.05), 0.1, 1.0)
        self.t += 1
        resp_time = self.load / (self.alloc + 1e-5)
        reward = -(resp_time * 2.0 + self.alloc * 1.0)
        return self._s(), float(reward), self.t >= 100

class ACNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.pi = nn.Linear(32, 3); self.v = nn.Linear(32, 1)
    def forward(self, x):
        h = self.fc(x); return torch.softmax(self.pi(h), -1), self.v(h)

def train(algo="A3C"):
    env = CloudEnv(); net = ACNet(); opt = optim.Adam(net.parameters(), lr=1e-3)
    rewards, resp_times = [], []
    for ep in range(150):
        s = env.reset(); lps, vals, rews = [], [], []
        done = False; ep_resp = []
        while not done:
            st = torch.FloatTensor(s)
            p, v = net(st)
            d = Categorical(p); a = d.sample()
            s, r, done = env.step(a.item())
            lps.append(d.log_prob(a)); vals.append(v); rews.append(r)
            ep_resp.append(s[0]/s[1])
        
        G, returns = 0, []
        for r in reversed(rews): G = r + 0.99*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        if algo == "VPG":
            loss = torch.stack([-lp * g for lp, g in zip(lps, returns)]).sum()
        else: # A3C / A2C
            v_tensor = torch.cat(vals).squeeze()
            adv = returns - v_tensor.detach()
            loss = -(torch.stack(lps) * adv).mean() + 0.5 * nn.functional.mse_loss(v_tensor, returns)
        opt.zero_grad(); loss.backward(); opt.step()
        rewards.append(sum(rews)); resp_times.append(np.mean(ep_resp))
    return rewards, resp_times

r_a3c, resp_a3c = train("A3C")
r_vpg, resp_vpg = train("VPG")

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(r_a3c, label="A3C"); plt.plot(r_vpg, label="VPG"); plt.title("Cloud Resource Allocation Reward"); plt.legend()
plt.subplot(122); plt.plot(resp_a3c, label="A3C"); plt.plot(resp_vpg, label="VPG"); plt.title("Average Response Time")
plt.tight_layout(); plt.show()
print(f"Final Reward -> A3C: {np.mean(r_a3c[-20:]):.2f}, VPG: {np.mean(r_vpg[-20:]):.2f}")
