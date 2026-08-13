"""
Q2: Markov Decision Process (MDP) Modeling
-------------------------------------------
Formal modeling of an MDP as a 5-tuple: M = (S, A, P, R, gamma)
- State space S = {S0, S1, S2}
- Action space A = {a0, a1}
- Transition probability tensor P(s' | s, a)
- Expected reward matrix R(s, a, s')
- Discount factor gamma = 0.9
"""

import numpy as np

class MarkovDecisionProcess:
    def __init__(self):
        self.states = ["S0_Idle", "S1_Working", "S2_Goal"]
        self.actions = ["a0_Rest", "a1_Work"]
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        self.gamma = 0.9
        
        # P[s, a, s'] = Probability of transitioning to s' given state s and action a
        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        # S0_Idle transitions
        self.P[0, 0, 0] = 1.0  # Rest in S0 -> stay S0
        self.P[0, 1, 1] = 0.8  # Work in S0 -> 80% to S1
        self.P[0, 1, 0] = 0.2  # Work in S0 -> 20% fail, stay S0
        
        # S1_Working transitions
        self.P[1, 0, 0] = 0.9  # Rest in S1 -> 90% drop to S0
        self.P[1, 0, 1] = 0.1  # Rest in S1 -> 10% stay S1
        self.P[1, 1, 2] = 0.7  # Work in S1 -> 70% reach S2 (Goal)
        self.P[1, 1, 1] = 0.3  # Work in S1 -> 30% stay S1
        
        # S2_Goal (Terminal state)
        self.P[2, 0, 2] = 1.0
        self.P[2, 1, 2] = 1.0
        
        # R[s, a, s'] = Expected reward
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R[0, 1, 1] = +2.0
        self.R[1, 1, 2] = +10.0
        self.R[1, 0, 0] = -1.0

    def verify_markov_property(self):
        """Verifies that for each (s, a), sum_s' P(s' | s, a) == 1.0."""
        valid = True
        for s in range(self.num_states):
            for a in range(self.num_actions):
                prob_sum = np.sum(self.P[s, a, :])
                if not np.isclose(prob_sum, 1.0):
                    valid = False
                    print(f"Validation Error: P(s' | s={s}, a={a}) sums to {prob_sum}")
        return valid

    def print_mdp_components(self):
        print("==================================================")
        print(" Q2: Markov Decision Process (MDP) Specification ")
        print("==================================================")
        print(f"States (S): {self.states}")
        print(f"Actions (A): {self.actions}")
        print(f"Discount Factor (gamma): {self.gamma}")
        print("\nTransition Probabilities P(s' | s, a):")
        for s, state_name in enumerate(self.states):
            for a, action_name in enumerate(self.actions):
                transitions = [
                    f"P({self.states[sp]}|{state_name},{action_name})={self.P[s,a,sp]:.2f}"
                    for sp in range(self.num_states) if self.P[s,a,sp] > 0
                ]
                print(f"  {state_name:<12} + {action_name:<10} -> {', '.join(transitions)}")

        print("\nVerification of Stochastic Property sum_s' P(s'|s,a) = 1:")
        print(f"  Passed: {self.verify_markov_property()}")
        print("==================================================\n")

if __name__ == "__main__":
    mdp = MarkovDecisionProcess()
    mdp.print_mdp_components()
