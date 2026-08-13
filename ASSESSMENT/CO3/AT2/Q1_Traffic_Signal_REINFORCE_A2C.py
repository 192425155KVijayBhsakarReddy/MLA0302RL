"""
QUESTION 1 (10 Marks):
Design and develop a Policy Gradient-based Reinforcement Learning model for an intelligent traffic signal control system.
Implement the solution using Python and TensorFlow/PyTorch, compare REINFORCE and A2C, and evaluate average vehicle
waiting time, throughput, and convergence rate.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Categorical

class TrafficEnv:
    def __init__(self): self.reset()
    def reset(self): self.q = np.random.randint(0, 5, 4).astype(float); self.phase = 0; self.step_c = 0; return self._get_s()
    def _get_s(self): return np.array([*self.q/20.0, self.phase], dtype=np.float32)
    def step(self, a):
        if a == 1: self.phase = 1 - self.phase
        self.q = np.clip(self.q + np.random.poisson(0.4, 4), 0, 20)
        c1, c2 = (0,1) if self.phase==0 else (2,3)
        clr = min(self.q[c1], 3) + min(self.q[c2], 3)
        self.q[c1] -= min(self.q[c1], 3); self.q[c2] -= min(self.q[c2], 3)
        self.step_c += 1
        return self._get_s(), float(clr*0.5 - np.sum(self.q)/80.0), self.step_c >= 100

class Net(nn.Module):
    def __init__(self, is_ac=False):
        super().__init__()
        self.is_ac = is_ac
        self.fc = nn.Sequential(nn.Linear(5, 32), nn.ReLU())
        self.pi = nn.Linear(32, 2)
        if is_ac: self.v = nn.Linear(32, 1)
    def forward(self, x):
        h = self.fc(x)
        return (torch.softmax(self.pi(h), -1), self.v(h)) if self.is_ac else torch.softmax(self.pi(h), -1)

def train(algo="REINFORCE"):
    env = TrafficEnv(); net = Net(is_ac=(algo=="A2C")); opt = optim.Adam(net.parameters(), lr=1e-2)
    rewards, waits, tpts = [], [], []
    for ep in range(150):
        s = env.reset(); lps, vals, r_list = [], [], []
        done = False; total_w = 0; total_t = 0
        while not done:
            st = torch.FloatTensor(s)
            out = net(st)
            probs, val = out if algo=="A2C" else (out, None)
            m = Categorical(probs); a = m.sample()
            s, r, done = env.step(a.item())
            lps.append(m.log_prob(a)); r_list.append(r)
            if algo=="A2C": vals.append(val)
            total_w += np.sum(env.q); total_t += (r>0)
        G, returns = 0, []
        for r in reversed(r_list): G = r + 0.99*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        if algo=="REINFORCE":
            loss = torch.stack([-lp * g for lp, g in zip(lps, returns)]).sum()
        else:
            vals = torch.cat(vals).squeeze()
            loss = (-(torch.stack(lps) * (returns - vals.detach())).mean() + nn.functional.mse_loss(vals, returns))
        opt.zero_grad(); loss.backward(); opt.step()
        rewards.append(sum(r_list)); waits.append(total_w/100); tpts.append(total_t)
    return rewards, waits, tpts

r_rf, w_rf, t_rf = train("REINFORCE")
r_a2c, w_a2c, t_a2c = train("A2C")
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.plot(r_rf, label="REINFORCE"); plt.plot(r_a2c, label="A2C"); plt.title("Reward"); plt.legend()
plt.subplot(132); plt.plot(w_rf, label="REINFORCE"); plt.plot(w_a2c, label="A2C"); plt.title("Avg Waiting Time")
plt.subplot(133); plt.plot(t_rf, label="REINFORCE"); plt.plot(t_a2c, label="A2C"); plt.title("Throughput")
plt.tight_layout(); plt.show()
print(f"REINFORCE Final Wait: {np.mean(w_rf[-20:]):.2f} | A2C Final Wait: {np.mean(w_a2c[-20:]):.2f}")
