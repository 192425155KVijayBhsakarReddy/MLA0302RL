"""
Q10: On-Policy (SARSA) vs Off-Policy (Q-Learning)
--------------------------------------------------
SARSA (On-Policy):   Q(s, a) <- Q(s, a) + alpha * [ R + gamma * Q(s', a') - Q(s, a) ]
                     where a' ~ behavior policy pi(s')

Q-Learning (Off-Policy): Q(s, a) <- Q(s, a) + alpha * [ R + gamma * max_a' Q(s', a') - Q(s, a) ]
                         evaluates greedy target policy while behaving with epsilon-greedy.

Cliff Walking Domain Comparison:
- SARSA learns a SAFE path away from the cliff (taking exploration risks into account).
- Q-Learning learns the OPTIMAL (shortest, risky) path along the cliff edge.
"""

import numpy as np

class CliffWalkingEnv:
    """
    1D Cliff Walking Grid: States 0 to 6.
    State 0: Start
    State 6: Goal (+10 reward)
    State 3: Cliff Trap (-100 reward, resets to State 0)
    Actions: 0 = Left, 1 = Right
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = 0
        return self.state
        
    def step(self, action):
        if action == 0:
            self.state = max(0, self.state - 1)
        else:
            self.state = min(6, self.state + 1)
            
        if self.state == 3: # Cliff
            reward = -100.0
            done = False
            self.state = 0 # Reset to start
        elif self.state == 6: # Goal
            reward = 10.0
            done = True
        else:
            reward = -1.0
            done = False
            
        return self.state, reward, done

def select_epsilon_greedy(Q, state, epsilon=0.2):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])
    return np.argmax(Q[state])

def train_sarsa(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    env = CliffWalkingEnv()
    Q = np.zeros((7, 2))
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        action = select_epsilon_greedy(Q, state, epsilon)
        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            next_state, reward, done = env.step(action)
            ep_reward += reward
            
            # SARSA target uses next_action from behavior policy
            next_action = select_epsilon_greedy(Q, next_state, epsilon)
            
            td_target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (td_target - Q[state, action])
            
            state = next_state
            action = next_action
            steps += 1
            
        rewards_history.append(ep_reward)
    return Q, rewards_history

def train_q_learning(episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    env = CliffWalkingEnv()
    Q = np.zeros((7, 2))
    rewards_history = []
    
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = select_epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            
            # Q-Learning target uses max_a' Q(s', a') regardless of behavior action
            best_next_q = np.max(Q[next_state])
            td_target = reward + gamma * best_next_q
            Q[state, action] += alpha * (td_target - Q[state, action])
            
            state = next_state
            steps += 1
            
        rewards_history.append(ep_reward)
    return Q, rewards_history

def main():
    print("==================================================")
    print(" Q10: On-Policy (SARSA) vs Off-Policy (Q-Learning) ")
    print("==================================================")
    
    episodes = 500
    Q_sarsa, r_sarsa = train_sarsa(episodes=episodes, epsilon=0.2)
    Q_qlearn, r_qlearn = train_q_learning(episodes=episodes, epsilon=0.2)
    
    print(f"Average Return over last 100 episodes (with exploration eps=0.2):")
    print(f"  SARSA (On-Policy):      {np.mean(r_sarsa[-100:]):.2f}")
    print(f"  Q-Learning (Off-Policy): {np.mean(r_qlearn[-100:]):.2f}\n")
    
    print("Key Operational Difference:")
    print("  - SARSA factors exploration mistakes into action values, leading to safer behavior online.")
    print("  - Q-Learning evaluates the greedy optimal target policy directly, despite exploring offline/online.")
    print("==================================================\n")

if __name__ == "__main__":
    main()
