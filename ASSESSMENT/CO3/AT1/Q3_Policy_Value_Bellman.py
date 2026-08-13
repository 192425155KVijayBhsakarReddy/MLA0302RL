"""
Q3: Policy, Value Functions, and Bellman Equations
--------------------------------------------------
Demonstrates policy evaluation using Bellman Expectation Equation:
V(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]
Q(s,a) = sum_s' P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]
"""

import numpy as np

class BellmanEvaluator:
    def __init__(self):
        self.states = [0, 1, 2, 3] # 3 is Goal State
        self.num_states = len(self.states)
        self.actions = [0, 1] # 0 = Left, 1 = Right
        self.num_actions = len(self.actions)
        self.gamma = 0.9
        
        # Policy pi(a|s): Uniform random policy (50% left, 50% right)
        self.policy = np.ones((self.num_states, self.num_actions)) * 0.5
        
    def get_transition_reward(self, s, a):
        """Returns list of (p, s_next, reward)."""
        if s == 3: # Terminal goal
            return [(1.0, 3, 0.0)]
            
        if a == 0: # Left
            s_next = max(0, s - 1)
            reward = 0.0
        else: # Right
            s_next = min(3, s + 1)
            reward = 10.0 if s_next == 3 else 0.0
            
        return [(1.0, s_next, reward)]

    def iterative_policy_evaluation(self, theta=1e-5):
        """Computes State-Value function V(s) using Bellman Expectation Equation."""
        V = np.zeros(self.num_states)
        
        while True:
            delta = 0
            for s in range(self.num_states):
                v_old = V[s]
                v_new = 0.0
                for a in range(self.num_actions):
                    pi_a_s = self.policy[s, a]
                    for prob, s_next, reward in self.get_transition_reward(s, a):
                        v_new += pi_a_s * prob * (reward + self.gamma * V[s_next])
                V[s] = v_new
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break
        return V

    def compute_action_value_function(self, V):
        """Computes Action-Value function Q(s, a) from state values V(s)."""
        Q = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                q_val = 0.0
                for prob, s_next, reward in self.get_transition_reward(s, a):
                    q_val += prob * (reward + self.gamma * V[s_next])
                Q[s, a] = q_val
        return Q

def main():
    print("==================================================")
    print(" Q3: Bellman Equation & Value Function Evaluation ")
    print("==================================================")
    
    evaluator = BellmanEvaluator()
    V = evaluator.iterative_policy_evaluation()
    Q = evaluator.compute_action_value_function(V)
    
    print("Computed State-Value Function V(s):")
    for s in range(evaluator.num_states):
        print(f"  V(State {s}) = {V[s]:.4f}")
        
    print("\nComputed Action-Value Function Q(s, a):")
    action_names = ["Left", "Right"]
    for s in range(evaluator.num_states):
        for a in range(evaluator.num_actions):
            print(f"  Q(State {s}, Action {action_names[a]}) = {Q[s, a]:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
