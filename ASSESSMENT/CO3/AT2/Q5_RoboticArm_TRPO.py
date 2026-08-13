"""
QUESTION 5 (10 Marks):
Implement a Trust Region Policy Optimization (TRPO) algorithm for controlling an Industrial robotic arm in an automated manufacturing environment.
Evaluate policy stability, precision, convergence speed, and overall production efficiency.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
from torch.distributions import Normal

class RoboticArmEnv:
    def __init__(self): self.reset()
    def reset(self): self.theta = 0.0; self.target = np.pi/2; self.t = 0; return self._s()
    def _s(self): return np.array([np.sin(self.theta), np.cos(self.theta), self.target], dtype=np.float32)
    def step(self, a):
        self.theta += np.clip(a, -0.2, 0.2)
        err = np.abs(self.theta - self.target)
        self.t += 1
        reward = -err + (1.0 if err < 0.05 else 0.0)
        return self._s(), float(reward), self.t >= 100

class TRPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.fc(x)

def train_trpo():
    env = RoboticArmEnv(); policy = TRPOPolicy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)
    rewards, precision_list = [], []
    for ep in range(120):
        s = env.reset(); states, actions, rews, log_probs_old = [], [], [], []
        done = False
        while not done:
            st = torch.FloatTensor(s)
            mu = policy(st)
            dist = Normal(mu, torch.tensor([0.1]))
            a = dist.sample()
            states.append(s); actions.append(a); log_probs_old.append(dist.log_prob(a))
            s, r, done = env.step(a.item())
            rews.append(r)
        
        G, returns = 0, []
        for r in reversed(rews): G = r + 0.98*G; returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        sts = torch.FloatTensor(np.array(states)); acts = torch.stack(actions)
        lps_old = torch.stack(log_probs_old).detach()
        for _ in range(5):
            mus = policy(sts); dists = Normal(mus, torch.tensor([0.1]))
            lps = dists.log_prob(acts)
            ratio = torch.exp(lps - lps_old)
            kl = (lps_old - lps).mean()
            if kl > 0.01: break  # Trust Region Constraint enforcement
            loss = -(ratio * returns).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            
        rewards.append(sum(rews)); precision_list.append(np.abs(env.theta - env.target))
    return rewards, precision_list

r_trpo, prec = train_trpo()
plt.figure(figsize=(10, 4))
plt.subplot(121); plt.plot(r_trpo); plt.title("TRPO Reward Convergence")
plt.subplot(122); plt.plot(prec); plt.title("Final Joint Angle Error (Precision)")
plt.tight_layout(); plt.show()
print(f"Final TRPO Reward: {np.mean(r_trpo[-20:]):.2f} | Final Precision Error: {prec[-1]:.4f} rad")
