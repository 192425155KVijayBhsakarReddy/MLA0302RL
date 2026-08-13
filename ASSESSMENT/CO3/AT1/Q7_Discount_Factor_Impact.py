"""
Q7: Discount Factor (gamma) Analysis
------------------------------------
Demonstrates how the discount factor gamma dictates an agent's effective horizon:
Effective Horizon = 1 / (1 - gamma)
- gamma = 0.0: Fully myopic (immediate reward only)
- gamma = 0.5: Short-term planning (effective horizon ~2 steps)
- gamma = 0.9: Balanced horizon (effective horizon ~10 steps)
- gamma = 0.99: Far-sighted planning (effective horizon ~100 steps)
"""

import numpy as np

class DeferredRewardEnv:
    """
    A 3-step decision task:
    Step 0: Choose Action A (Small Immediate Reward +1, but max future reward +5)
            OR Choose Action B (Immediate Reward 0, but massive delayed reward +100 at Step 3).
    """
    def __init__(self):
        pass
        
    def evaluate_trajectory_return(self, choice, gamma):
        if choice == "A":
            rewards = [1.0, 5.0, 5.0]
        else: # choice == "B"
            rewards = [0.0, 0.0, 100.0]
            
        G_0 = sum((gamma ** t) * r for t, r in enumerate(rewards))
        return G_0

def main():
    print("==================================================")
    print(" Q7: Discount Factor (gamma) Horizon Evaluation ")
    print("==================================================")
    
    env = DeferredRewardEnv()
    gammas = [0.0, 0.5, 0.8, 0.9, 0.99]
    
    print(f"{'Gamma (gamma)':<15}{'Effective Horizon':<20}{'Return G(Choice A)':<22}{'Return G(Choice B)':<22}{'Optimal Preference':<20}")
    print("-" * 95)
    
    for gamma in gammas:
        horizon = "1 step (Myopic)" if gamma == 0 else f"~{1/(1-gamma):.1f} steps"
        ret_A = env.evaluate_trajectory_return("A", gamma)
        ret_B = env.evaluate_trajectory_return("B", gamma)
        
        pref = "Choice A (Short-term)" if ret_A > ret_B else "Choice B (Long-term)"
        print(f"{gamma:<15.2f}{horizon:<20}{ret_A:<22.3f}{ret_B:<22.3f}{pref:<20}")
        
    print("-" * 95)
    print("Key Insight:")
    print("  Low gamma (< 0.8) makes the agent prefer immediate gratification (Choice A).")
    print("  High gamma (>= 0.8) empowers the agent to endure short-term zero rewards for huge long-term payouts (Choice B).")
    print("==================================================\n")

if __name__ == "__main__":
    main()
