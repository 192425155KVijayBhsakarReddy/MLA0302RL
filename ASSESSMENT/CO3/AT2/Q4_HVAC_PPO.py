"""
QUESTION 4 (10 Marks):
Design and develop a Proximal Policy Optimization (PPO) model for controlling Smart HVAC Energy Management systems in a smart building.
Implement the model to minimize energy consumption while maintaining occupant comfort, and compare PPO with REINFORCE.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Normal

class HVACEnv:
    def __init__(self): self.reset()
    def reset(self): self.temp = 25.0; self.out_temp = 32.0; self.t = 0; return self._s()
    def _s(self): return np.array([(self.temp-22.0)/10.0, (self.out_temp-30.0)/10.0], dtype=np.float32)
    def step(self, a):
        cooling = np.clip(a, 0, 5)
        self.temp += 0.1 * (self.out_temp - self.temp) - 0.3 * cooling
        self.t += 1
        energy_cost = 0.5 * (cooling ** 2)
        comfort_penalty = 2.0 * ((self.temp - 22.0) ** 2)
        reward = -(energy_cost + comfort_penalty)
        return self._s(), float(reward), self.t >= 100

class PPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.mu = nn.Linear(32, 1); self.v = nn.Linear(32, 1)
    def forward(self, x):
        h = self.fc(x); return self.mu(h), self.v(h)

def train(algo="PPO"):
    env = HVACEnv(); net = PPONet(); opt = optim.Adam(net.parameters(), lr=1e-3)
    rewards, energy_list = [], []
    for ep in range(150):
        s = env.reset(); states, actions, rews, log_probs_old = [], [], [], []
        done = False
        while not done:
            st = torch.FloatTensor(s)
            mu, _ = net(st)
            dist = Normal(mu, torch.tensor([0.5]))
            a = dist.sample(); lp = dist.log_prob(a)
            states.append(s); actions.append(a); log_probs_old.append(lp);
            s, r, done = env.step(a.item())
            rews.append(r)
        
        G, returns = 0, []
        for r in reversed(rews): G = r + 0.95*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        if algo == "REINFORCE":
            loss = torch.stack([-lp * g for lp, g in zip(log_probs_old, returns)]).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        else: # PPO clipped surrogate
            sts = torch.FloatTensor(np.array(states))
            acts = torch.stack(actions); lps_old = torch.stack(log_probs_old).detach()
            for _ in range(4):
                mus, vals = net(sts)
                dists = Normal(mus, torch.tensor([0.5]))
                lps = dists.log_prob(acts)
                ratios = torch.exp(lps - lps_old)
                advs = (returns - vals.squeeze().detach()).unsqueeze(1)
                surr1 = ratios * advs
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advs
                loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.functional.mse_loss(vals.squeeze(), returns)
                opt.zero_grad(); loss.backward(); opt.step()

        rewards.append(sum(rews)); energy_list.append(-sum(rews))
    return rewards, energy_list

r_ppo, e_ppo = train("PPO")
r_rf, e_rf = train("REINFORCE")

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(r_ppo, label="PPO"); plt.plot(r_rf, label="REINFORCE"); plt.title("Cumulative Reward"); plt.legend()
plt.subplot(122); plt.plot(e_ppo, label="PPO"); plt.plot(e_rf, label="REINFORCE"); plt.title("Energy + Discomfort Cost")
plt.tight_layout(); plt.show()
print(f"Final Rewards -> PPO: {np.mean(r_ppo[-20:]):.2f}, REINFORCE: {np.mean(r_rf[-20:]):.2f}")
