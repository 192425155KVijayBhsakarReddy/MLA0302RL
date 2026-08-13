"""
Q1: Reinforcement Learning (RL) Framework Implementation
---------------------------------------------------------
Demonstrates the agent-environment interaction loop:
- State Space S, Action Space A, Reward Function R
- Step-by-step state transition
- Calculation of cumulative return (discounted sum of rewards G_t)
"""

import random

class SimpleGridEnvironment:
    """
    A 1D Grid Environment with states [0, 1, 2, 3, 4].
    State 4 is the Goal state (+10 reward).
    State 0 is a trap (-5 reward).
    Actions: 0 = Move Left, 1 = Move Right.
    """
    def __init__(self):
        self.num_states = 5
        self.reset()
        
    def reset(self):
        self.current_state = 2  # Start in the middle
        return self.current_state
        
    def step(self, action):
        # Action 0: Left, Action 1: Right
        if action == 0:
            self.current_state = max(0, self.current_state - 1)
        elif action == 1:
            self.current_state = min(self.num_states - 1, self.current_state + 1)
            
        # Determine reward and termination
        if self.current_state == 4:
            reward = 10.0
            done = True
        elif self.current_state == 0:
            reward = -5.0
            done = True
        else:
            reward = -0.1  # Step penalty to encourage efficiency
            done = False
            
        return self.current_state, reward, done

class RandomAgent:
    """Agent that interacts with the environment by selecting random actions."""
    def select_action(self, state):
        return random.choice([0, 1])

def calculate_cumulative_return(rewards, gamma=0.95):
    """Calculates cumulative discounted return G_t for an episode trajectory."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def main():
    print("==================================================")
    print(" Q1: Agent-Environment Interaction Simulation ")
    print("==================================================")
    
    env = SimpleGridEnvironment()
    agent = RandomAgent()
    gamma = 0.95
    
    state = env.reset()
    trajectory = []
    rewards = []
    done = False
    step_count = 0
    
    print(f"Initial State: s_0 = {state}\n")
    print(f"{'Step':<6}{'State (s_t)':<14}{'Action (a_t)':<14}{'Next State (s_{t+1})':<20}{'Reward (r_{t+1})':<15}")
    print("-" * 70)
    
    while not done and step_count < 15:
        action = agent.select_action(state)
        action_name = "Left" if action == 0 else "Right"
        next_state, reward, done = env.step(action)
        
        trajectory.append((state, action_name, reward, next_state))
        rewards.append(reward)
        
        print(f"{step_count:<6}{state:<14}{action_name:<14}{next_state:<20}{reward:<15.2f}")
        state = next_state
        step_count += 1

    discounted_returns = calculate_cumulative_return(rewards, gamma=gamma)
    
    print("-" * 70)
    print(f"Total Trajectory Rewards: {sum(rewards):.2f}")
    print(f"Initial Cumulative Discounted Return G_0 (gamma={gamma}): {discounted_returns[0]:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
