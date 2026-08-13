"""
QUESTION 9 (10 Marks):
Develop a Policy Gradient-based predictive maintenance system for industrial machines.
Design the RL environment, reward function, and policy network to minimize maintenance costs and machine downtime.
Compare the performance of REINFORCE and PPO.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Categorical

class MaintenanceEnv:
    def __init__(self): self.reset()
    def reset(self): self.wear = 0.0; self.t = 0; return self._s()
    def _s(self): return np.array([self.wear/100.0, self.t/100.0], dtype=np.float32)
    def step(self, a):
        cost = 0.0
        if a == 0:
            self.wear += np.random.uniform(2.0, 8.0)
            if self.wear >= 100.0: cost = 50.0  # Breakdown failure penalty
        elif a == 1:
            self.wear = max(0.0, self.wear - 30.0); cost = 5.0
        elif a == 2:
            self.wear = 0.0; cost = 15.0
        self.t += 1
        reward = -(cost + (2.0 if self.wear > 80.0 else 0.0))
        return self._s(), float(reward), self.t >= 100 or self.wear >= 100.0

class MaintenanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.pi = nn.Linear(32, 3); self.v = nn.Linear(32, 1)
    def forward(self, x):
        h = self.fc(x); return torch.softmax(self.pi(h), -1), self.v(h)

def train(algo="PPO"):
    env = MaintenanceEnv(); net = MaintenanceNet(); opt = optim.Adam(net.parameters(), lr=1e-3)
    rewards, breakdowns = [], []
    for ep in range(150):
        s = env.reset(); states, actions, lps, rews = [], [], [], []
        done = False; bd = 0
        while not done:
            st = torch.FloatTensor(s)
            p, v = net(st)
            d = Categorical(p); a = d.sample()
            states.append(s); actions.append(a); lps.append(d.log_prob(a))
            s, r, done = env.step(a.item())
            rews.append(r)
            if env.wear >= 100.0: bd = 1
        
        G, returns = 0, []
        for r in reversed(rews): G = r + 0.99*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        if algo == "REINFORCE":
            loss = torch.stack([-lp * g for lp, g in zip(lps, returns)]).sum()
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
                
        rewards.append(sum(rews)); breakdowns.append(bd)
    return rewards, breakdowns

r_ppo, bd_ppo = train("PPO")
r_rf, bd_rf = train("REINFORCE")

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(r_ppo, label="PPO"); plt.plot(r_rf, label="REINFORCE"); plt.title("Maintenance Cost Minimization Reward"); plt.legend()
plt.subplot(122); plt.plot(bd_ppo, label="PPO"); plt.plot(bd_rf, label="REINFORCE"); plt.title("Machine Breakdown Events")
plt.tight_layout(); plt.show()
print(f"Final Reward -> PPO: {np.mean(r_ppo[-20:]):.2f}, REINFORCE: {np.mean(r_rf[-20:]):.2f}")
