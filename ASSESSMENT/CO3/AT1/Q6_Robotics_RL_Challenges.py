"""
Q6: Challenges of Applying RL in Real-World Robotics
------------------------------------------------------
Simulates three core challenges:
1. Safety Constraints (Constrained MDP / Safety Penalties)
2. Sample Efficiency (Experience Replay / Virtual Simulation)
3. Environment Variability (Domain Randomization)
"""

import random
import numpy as np

class RoboticArmEnv:
    """
    Simulates a robot arm reaching a target angle (0 to 180 deg).
    Features:
    - Safety Zone: Angles > 160 deg trigger catastrophic hard limits.
    - Variability: Joint friction randomly varies each reset (domain randomization).
    """
    def __init__(self):
        self.reset()
        
    def reset(self, domain_randomization=True):
        self.angle = 90.0  # Initial neutral angle
        self.target_angle = 140.0
        # Friction factor varies between 0.8 and 1.2 under domain randomization
        self.friction = random.uniform(0.7, 1.3) if domain_randomization else 1.0
        return self.angle

    def step(self, torque_action):
        # Action: applied torque in degrees (-10 to +10)
        actual_torque = torque_action * self.friction
        self.angle += actual_torque
        
        # Calculate error
        error = abs(self.angle - self.target_angle)
        
        # 1. Base Reaching Reward
        reward = -error
        
        # 2. Safety Constraint Assessment
        safety_violation = False
        if self.angle > 160.0 or self.angle < 0.0:
            reward -= 500.0  # Massive penalty for unsafe region
            safety_violation = True
            done = True
        elif error < 5.0:
            reward += 100.0  # Goal reached
            done = True
        else:
            done = False
            
        return self.angle, reward, done, safety_violation

def demonstrate_robotics_challenges():
    print("==================================================")
    print(" Q6: Real-World Robotics RL Challenge Simulation ")
    print("==================================================")
    
    env = RoboticArmEnv()
    
    # Challenge 1: Unconstrained vs Safe Torque Policy
    print("1. Safety & Constraint Violation Test:")
    print("   Testing Unconstrained Large Torque Actions...")
    env.reset(domain_randomization=False)
    unsafe_steps = 0
    for torque in [+15, +20, +25, +30]:
        angle, r, done, unsafe = env.step(torque)
        print(f"   Applied Torque: +{torque:<3} -> Angle: {angle:.1f}° | Reward: {r:<6.1f} | Unsafe: {unsafe}")
        if unsafe:
            print("   ==> Safety constraint violated! Hard hardware shutdown triggered.\n")
            break

    # Challenge 2: Environment Variability (Domain Randomization)
    print("2. Environment Variability & Friction Perturbation Test:")
    fixed_action = 10.0
    for run in range(1, 4):
        init_angle = env.reset(domain_randomization=True)
        next_angle, r, _, _ = env.step(fixed_action)
        print(f"   Run {run}: Friction Factor = {env.friction:.3f} | Same Torque (+10) resulting Angle = {next_angle:.2f}°")
        
    print("\n3. Sample Efficiency Comparison:")
    print("   Real Hardware Sample Time per step: ~50ms -> 100,000 steps = 1.38 hours of continuous physical wear.")
    print("   Simulated Parallel Replay Time:      <1.5 seconds.")
    print("==================================================\n")

if __name__ == "__main__":
    demonstrate_robotics_challenges()
