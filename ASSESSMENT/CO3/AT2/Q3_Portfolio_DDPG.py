"""
QUESTION 3 (10 Marks):
Develop a Deep Deterministic Policy Gradient (DDPG) model for continuous portfolio optimization in stock trading.
Design the state space, action space, reward function, and evaluate cumulative return, risk, and convergence performance.
"""

import numpy as np, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt, random

class PortfolioEnv:
    def __init__(self, n=4): self.n = n; self.reset()
    def reset(self): self.w = np.ones(self.n)/self.n; self.t = 0; self.val = 1.0; self.vals = [1.0]; return self._s()
    def _s(self): return np.concatenate([self.w, [self.val]]).astype(np.float32)
    def step(self, a):
        w = np.exp(a) / (np.sum(np.exp(a)) + 1e-8)
        rets = np.random.normal(0.0005, 0.015, self.n)
        r_p = np.dot(w, rets) - 0.001 * np.sum(np.abs(w - self.w))
        self.val *= (1.0 + r_p); self.w = w; self.t += 1
        self.vals.append(self.val)
        return self._s(), float(r_p - 0.1 * np.var(rets)), self.t >= 100

class Actor(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n+1, 32), nn.ReLU(), nn.Linear(32, n), nn.Tanh())
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*n+1, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, s, a): return self.net(torch.cat([s, a], -1))

def train_ddpg():
    env = PortfolioEnv(); act = Actor(); crt = Critic()
    opt_a = optim.Adam(act.parameters(), lr=1e-3); opt_c = optim.Adam(crt.parameters(), lr=1e-3)
    rewards, cum_returns, sharpes = [], [], []
    for ep in range(120):
        s = env.reset(); ep_r = 0; done = False
        while not done:
            st = torch.FloatTensor(s)
            a = act(st).detach().numpy() + np.random.normal(0, 0.1, env.n)
            s2, r, done = env.step(a)
            st2 = torch.FloatTensor(s2); at = torch.FloatTensor(a)
            target = r + 0.99 * crt(st2, act(st2)).item() * (1 - float(done))
            c_loss = nn.functional.mse_loss(crt(st, at), torch.tensor([target]))
            opt_c.zero_grad(); c_loss.backward(); opt_c.step()
            a_loss = -crt(st, act(st)).mean()
            opt_a.zero_grad(); a_loss.backward(); opt_a.step()
            s = s2; ep_r += r
        rewards.append(ep_r); cum_returns.append((env.val - 1.0)*100)
        arr = np.diff(env.vals); sharpes.append((arr.mean()/(arr.std()+1e-8))*np.sqrt(252))
    return rewards, cum_returns, sharpes

r, cr, sh = train_ddpg()
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.plot(r); plt.title("Episode Reward (Convergence)")
plt.subplot(132); plt.plot(cr); plt.title("Cumulative Return (%)")
plt.subplot(133); plt.plot(sh); plt.title("Sharpe Ratio (Risk-Adjusted)")
plt.tight_layout(); plt.show()
print(f"Final Return: {cr[-1]:.2f}% | Final Sharpe: {sh[-1]:.2f}")
