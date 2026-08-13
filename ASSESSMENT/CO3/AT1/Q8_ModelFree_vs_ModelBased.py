"""
Q8: Model-Free vs. Model-Based Reinforcement Learning
------------------------------------------------------
Compares:
1. Model-Free RL (Standard Q-Learning)
2. Model-Based RL (Dyna-Q with N planning steps)
In a Gridworld environment to show sample efficiency differences.
"""

import random
import numpy as np

class GridWorld:
    def __init__(self, width=6, height=6):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (5, 5)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        
    def step(self, state, action_idx):
        dr, dc = self.actions[action_idx]
        nr, nc = state[0] + dr, state[1] + dc
        nr = max(0, min(self.height - 1, nr))
        nc = max(0, min(self.width - 1, nc))
        next_state = (nr, nc)
        
        reward = 100.0 if next_state == self.goal else -1.0
        done = (next_state == self.goal)
        return next_state, reward, done

def run_dynaq(planning_steps=0, episodes=20, alpha=0.1, gamma=0.95):
    env = GridWorld()
    Q = {}
    Model = {}
    
    def get_q(s, a):
        return Q.get((s, a), 0.0)

    steps_history = []

    for ep in range(episodes):
        state = env.start
        done = False
        steps = 0
        
        while not done and steps < 300:
            # Epsilon-greedy action choice
            if random.random() < 0.1:
                action_idx = random.randint(0, 3)
            else:
                q_vals = [get_q(state, a) for a in range(4)]
                action_idx = np.argmax(q_vals)
                
            next_state, reward, done = env.step(state, action_idx)
            
            # Direct Reinforcement Learning update (Model-Free part)
            best_next_q = max([get_q(next_state, a) for a in range(4)])
            old_q = get_q(state, action_idx)
            Q[(state, action_idx)] = old_q + alpha * (reward + gamma * best_next_q - old_q)
            
            # Model Learning (Model-Based part)
            Model[(state, action_idx)] = (reward, next_state)
            
            # Planning phase (Dyna-Q)
            for _ in range(planning_steps):
                # Random previously observed state and action
                (s_p, a_p), (r_p, ns_p) = random.choice(list(Model.items()))
                best_ns_q = max([get_q(ns_p, a) for a in range(4)])
                Q[(s_p, a_p)] += alpha * (r_p + gamma * best_ns_q - Q[(s_p, a_p)])
                
            state = next_state
            steps += 1
            
        steps_history.append(steps)
        
    return steps_history

def main():
    print("==================================================")
    print(" Q8: Model-Free (Q-Learning) vs Model-Based (Dyna-Q)")
    print("==================================================")
    
    episodes = 15
    # Model-Free (Planning steps = 0)
    steps_mf = run_dynaq(planning_steps=0, episodes=episodes)
    
    # Model-Based Dyna-Q (Planning steps = 10 simulated steps per real step)
    steps_mb = run_dynaq(planning_steps=10, episodes=episodes)
    
    print(f"{'Episode':<10}{'Model-Free Steps (Q-Learning)':<32}{'Model-Based Steps (Dyna-Q, N=10)':<32}")
    print("-" * 75)
    for ep in range(episodes):
        print(f"{ep+1:<10}{steps_mf[ep]:<32}{steps_mb[ep]:<32}")
        
    print("-" * 75)
    print(f"Total steps required over {episodes} episodes:")
    print(f"  Model-Free (Q-Learning): {sum(steps_mf)} environment steps")
    print(f"  Model-Based (Dyna-Q):    {sum(steps_mb)} environment steps")
    print("Conclusion: Model-Based RL achieves drastically higher sample efficiency via internal model planning.")
    print("==================================================\n")

if __name__ == "__main__":
    main()
