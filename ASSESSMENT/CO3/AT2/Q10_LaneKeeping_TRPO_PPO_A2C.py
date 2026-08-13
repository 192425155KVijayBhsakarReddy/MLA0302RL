"""
QUESTION 10 (10 Marks):
Design, implement, and evaluate a Policy-Based Reinforcement Learning controller for autonomous lane-keeping using TRPO or PPO.
Compare the selected algorithm with A2C based on lane deviation, safety, reward convergence, and training efficiency.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Normal

class LaneKeepingEnv:
    def __init__(self): self.reset()
    def reset(self): self.y = 0.0; self.angle = 0.0; self.t = 0; return self._s()
    def _s(self): return np.array([self.y/2.0, self.angle/0.5], dtype=np.float32)
    def step(self, a):
        steer = np.clip(a, -0.1, 0.1)
        self.angle += steer
        self.y += np.sin(self.angle)
        self.t += 1
        dev = np.abs(self.y)
        reward = 1.0 - dev - 5.0 * np.abs(steer)
        return self._s(), float(reward), self.t >= 100 or dev > 2.0

class LaneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.mu = nn.Linear(32, 1); self.v = nn.Linear(32, 1)
    def forward(self, x):
        h = self.fc(x); return torch.tanh(self.mu(h))*0.1, self.v(h)

def train(algo="PPO"):
    env = LaneKeepingEnv(); net = LaneNet(); opt = optim.Adam(net.parameters(), lr=1e-3)
    rewards, devs = [], []
    for ep in range(150):
        s = env.reset(); states, actions, lps, rews = [], [], [], []
        done = False
        while not done:
            st = torch.FloatTensor(s)
            mu, v = net(st)
            dist = Normal(mu, torch.tensor([0.02]))
            a = dist.sample()
            states.append(s); actions.append(a); lps.append(dist.log_prob(a))
            s, r, done = env.step(a.item())
            rews.append(r)
        
        G, returns = 0, []
        for r in reversed(rews): G = r + 0.98*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        if algo == "A2C":
            sts = torch.FloatTensor(np.array(states))
            _, v_out = net(sts)
            adv = returns - v_out.squeeze().detach()
            loss = -(torch.stack(lps) * adv).mean() + 0.5 * nn.functional.mse_loss(v_out.squeeze(), returns)
            opt.zero_grad(); loss.backward(); opt.step()
        else: # PPO / TRPO
            sts = torch.FloatTensor(np.array(states)); acts = torch.stack(actions); lps_old = torch.stack(lps).detach()
            for _ in range(4):
                mus, v_out = net(sts); dists = Normal(mus, torch.tensor([0.02]))
                lps_new = dists.log_prob(acts)
                ratio = torch.exp(lps_new - lps_old)
                advs = returns - v_out.squeeze().detach()
                loss = -torch.min(ratio * advs, torch.clamp(ratio, 0.8, 1.2) * advs).mean() + 0.5 * nn.functional.mse_loss(v_out.squeeze(), returns)
                opt.zero_grad(); loss.backward(); opt.step()
                
        rewards.append(sum(rews)); devs.append(np.abs(env.y))
    return rewards, devs

r_ppo, d_ppo = train("PPO")
r_a2c, d_a2c = train("A2C")

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(r_ppo, label="PPO/TRPO"); plt.plot(r_a2c, label="A2C"); plt.title("Lane-Keeping Reward Convergence"); plt.legend()
plt.subplot(122); plt.plot(d_ppo, label="PPO/TRPO"); plt.plot(d_a2c, label="A2C"); plt.title("Final Lane Deviation (m)")
plt.tight_layout(); plt.show()
print(f"Final Lane Deviation -> PPO/TRPO: {d_ppo[-1]:.4f}m, A2C: {d_a2c[-1]:.4f}m")
