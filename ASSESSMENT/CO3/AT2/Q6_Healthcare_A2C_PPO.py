"""
QUESTION 6 (10 Marks):
Design and develop a Policy-Based Reinforcement Learning system for Healthcare Treatment adaptive patient treatment planning.
Compare A2C and PPO in terms of treatment effectiveness, cumulative reward, and learning stability using simulated patient data.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Categorical

class PatientEnv:
    def __init__(self): self.reset()
    def reset(self): self.health = 50.0; self.tumor = 50.0; self.t = 0; return self._s()
    def _s(self): return np.array([self.health/100.0, self.tumor/100.0], dtype=np.float32)
    def step(self, a):
        dosage = a * 10.0
        self.tumor = max(0.0, self.tumor - 1.5 * dosage + 2.0)
        self.health = max(0.0, self.health - 0.5 * dosage + 1.0)
        self.t += 1
        reward = (50.0 - self.tumor) + (self.health - 50.0) - 0.1 * dosage
        return self._s(), float(reward), self.t >= 60 or self.health <= 0

class HealthcareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.pi = nn.Linear(32, 3); self.v = nn.Linear(32, 1)
    def forward(self, x):
        h = self.fc(x); return torch.softmax(self.pi(h), -1), self.v(h)

def train(algo="A2C"):
    env = PatientEnv(); net = HealthcareNet(); opt = optim.Adam(net.parameters(), lr=1e-3)
    rewards, health_hist = [], []
    for ep in range(150):
        s = env.reset(); states, actions, lps, vals, rews = [], [], [], [], []
        done = False
        while not done:
            st = torch.FloatTensor(s)
            p, v = net(st)
            d = Categorical(p); a = d.sample()
            states.append(s); actions.append(a); lps.append(d.log_prob(a)); vals.append(v); rews.append(0)
            s, r, done = env.step(a.item())
            rews[-1] = r
        
        G, returns = 0, []
        for r in reversed(rews): G = r + 0.99*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        if algo == "A2C":
            v_tensor = torch.cat(vals).squeeze()
            adv = returns - v_tensor.detach()
            loss = -(torch.stack(lps) * adv).mean() + 0.5 * nn.functional.mse_loss(v_tensor, returns)
            opt.zero_grad(); loss.backward(); opt.step()
        else: # PPO
            sts = torch.FloatTensor(np.array(states)); acts = torch.stack(actions); lps_old = torch.stack(lps).detach()
            for _ in range(4):
                probs, v_out = net(sts); dists = Categorical(probs)
                lps_new = dists.log_prob(acts)
                ratio = torch.exp(lps_new - lps_old)
                advs = returns - v_out.squeeze().detach()
                loss = -torch.min(ratio * advs, torch.clamp(ratio, 0.8, 1.2) * advs).mean() + 0.5 * nn.functional.mse_loss(v_out.squeeze(), returns)
                opt.zero_grad(); loss.backward(); opt.step()
                
        rewards.append(sum(rews)); health_hist.append(env.health)
    return rewards, health_hist

r_a2c, h_a2c = train("A2C")
r_ppo, h_ppo = train("PPO")

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(r_a2c, label="A2C"); plt.plot(r_ppo, label="PPO"); plt.title("Cumulative Reward"); plt.legend()
plt.subplot(122); plt.plot(h_a2c, label="A2C"); plt.plot(h_ppo, label="PPO"); plt.title("Final Patient Health")
plt.tight_layout(); plt.show()
print(f"Final Reward -> A2C: {np.mean(r_a2c[-20:]):.2f}, PPO: {np.mean(r_ppo[-20:]):.2f}")
