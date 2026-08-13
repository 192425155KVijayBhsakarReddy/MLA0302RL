"""
Q5: Reward Shaping and Policy Acceleration
-------------------------------------------
Demonstrates Potential-Based Reward Shaping (Ng et al., 1999):
Shaped Reward: R'(s, a, s') = R(s, a, s') + F(s, s')
Where Potential Difference F(s, s') = gamma * Phi(s') - Phi(s)
Guarantees optimal policy invariance while accelerating convergence.
"""

import numpy as np

class SparseGridworld:
    """1D Gridworld where goal is at index 9. Base reward is 0 everywhere except +100 at Goal."""
    def __init__(self, size=10):
        self.size = size
        self.goal = size - 1
        
    def reset(self):
        return 0
        
    def step(self, state, action):
        # Action 0: Left (-1), Action 1: Right (+1)
        next_state = state - 1 if action == 0 else state + 1
        next_state = max(0, min(self.size - 1, next_state))
        
        base_reward = 100.0 if next_state == self.goal else 0.0
        done = (next_state == self.goal)
        return next_state, base_reward, done

def potential_function(state, goal_state=9):
    """Potential function Phi(s) proportional to closeness to goal."""
    return 10.0 * (state / goal_state)

def run_q_learning(use_shaping=False, episodes=100, alpha=0.1, gamma=0.9):
    env = SparseGridworld(size=10)
    Q = np.zeros((10, 2))
    steps_per_episode = []
    
    for ep in range(episodes):
        state = env.reset()
        steps = 0
        done = False
        
        while not done and steps < 200:
            # Epsilon greedy selection
            if np.random.rand() < 0.1:
                action = np.random.choice([0, 1])
            else:
                action = np.argmax(Q[state])
                
            next_state, base_reward, done = env.step(state, action)
            
            # Apply potential-based reward shaping if enabled
            if use_shaping:
                phi_s = potential_function(state)
                phi_s_next = potential_function(next_state)
                F = gamma * phi_s_next - phi_s
                shaped_reward = base_reward + F
            else:
                shaped_reward = base_reward
                
            # Q-learning update
            Q[state, action] += alpha * (shaped_reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            steps += 1
            
        steps_per_episode.append(steps)
        
    return steps_per_episode, Q

def main():
    print("==================================================")
    print(" Q5: Potential-Based Reward Shaping Acceleration ")
    print("==================================================")
    
    episodes = 50
    # Standard Sparse Reward RL
    steps_unshaped, Q_unshaped = run_q_learning(use_shaping=False, episodes=episodes)
    
    # Potential-Based Reward Shaped RL
    steps_shaped, Q_shaped = run_q_learning(use_shaping=True, episodes=episodes)
    
    print(f"Average Steps to Goal over first 10 episodes:")
    print(f"  Unshaped Sparse Reward: {np.mean(steps_unshaped[:10]):.2f} steps")
    print(f"  Shaped Reward:          {np.mean(steps_shaped[:10]):.2f} steps\n")
    
    print(f"Average Steps to Goal over last 10 episodes:")
    print(f"  Unshaped Sparse Reward: {np.mean(steps_unshaped[-10:]):.2f} steps")
    print(f"  Shaped Reward:          {np.mean(steps_shaped[-10:]):.2f} steps\n")
    
    print("Final Optimal Action Choices (State 0 to 8):")
    print("Unshaped Policy (0=Left, 1=Right):", np.argmax(Q_unshaped[:9], axis=1))
    print("Shaped Policy   (0=Left, 1=Right):", np.argmax(Q_shaped[:9], axis=1))
    print("Notice: Both converge to the identical optimal policy (all 1s), but shaping speeds up learning!")
    print("==================================================\n")

if __name__ == "__main__":
    main()
