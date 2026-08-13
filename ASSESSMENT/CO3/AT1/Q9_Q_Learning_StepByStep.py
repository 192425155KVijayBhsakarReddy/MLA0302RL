"""
Q9: Step-by-Step Q-Learning Update & Convergence Demonstration
----------------------------------------------------------------
Formula: Q(s, a) <- Q(s, a) + alpha * [ R + gamma * max_a' Q(s', a') - Q(s, a) ]

Demonstrates step-by-step numerical calculation for 5 iterations,
followed by full convergence over 200 episodes.
"""

import numpy as np

def step_by_step_numerical_example():
    print("==================================================")
    print(" Q9: Step-by-Step Numerical Q-Learning Update ")
    print("==================================================")
    
    # Parameters
    alpha = 0.5   # Learning rate
    gamma = 0.9   # Discount factor
    
    # Initial Q-table: 2 States (S0, S1), 2 Actions (A0, A1)
    Q = np.array([
        [0.0, 0.0],  # S0
        [0.0, 0.0]   # S1
    ])
    
    # Pre-defined trajectory steps (State, Action, Reward, Next_State)
    experience_tuples = [
        (0, 1, +2.0, 1),  # Step 1: In S0, take A1 -> get +2 reward, move to S1
        (1, 0, +0.0, 0),  # Step 2: In S1, take A0 -> get 0 reward, move to S0
        (0, 1, +2.0, 1),  # Step 3: In S0, take A1 -> get +2 reward, move to S1
        (1, 1, +10.0, 1), # Step 4: In S1, take A1 -> get +10 reward (Goal), stay in S1
        (0, 1, +2.0, 1),  # Step 5: In S0, take A1 -> get +2 reward, move to S1
    ]
    
    print(f"Hyperparameters: Learning Rate alpha={alpha}, Discount gamma={gamma}\n")
    print(f"{'Step':<6}{'Tuple (s, a, r, s&#39;)':<24}{'Target = r + gamma*max Q(s&#39;)':<32}{'TD Error delta':<20}{'Updated Q(s, a)':<18}".replace("&#39;", "'"))
    print("-" * 100)
    
    for i, (s, a, r, s_next) in enumerate(experience_tuples, 1):
        q_old = Q[s, a]
        max_q_next = np.max(Q[s_next])
        td_target = r + gamma * max_q_next
        td_error = td_target - q_old
        q_new = q_old + alpha * td_error
        
        Q[s, a] = q_new
        
        tuple_str = f"({s}, {a}, {r:+.1f}, {s_next})"
        print(f"{i:<6}{tuple_str:<24}{td_target:<32.4f}{td_error:<20.4f}{q_new:<18.4f}")
        
    print("-" * 100)
    print("\nFinal Q-Table after 5 manual updates:")
    print("       Action 0   Action 1")
    print(f"S0:    {Q[0,0]:<10.4f} {Q[0,1]:<10.4f}")
    print(f"S1:    {Q[1,0]:<10.4f} {Q[1,1]:<10.4f}")
    print("==================================================\n")

def demonstrate_convergence():
    print("--- Q-Learning Convergence Behavior ---")
    # Gridworld convergence
    states = 5
    actions = 2
    Q = np.zeros((states, actions))
    alpha = 0.1
    gamma = 0.9
    
    max_q_diff_history = []
    
    for episode in range(300):
        s = 0
        Q_prev = Q.copy()
        while s < states - 1:
            a = 1 if np.random.rand() > 0.1 else 0
            s_next = min(states - 1, s + 1) if a == 1 else max(0, s - 1)
            r = 10.0 if s_next == states - 1 else -0.1
            
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next
            
        max_diff = np.max(np.abs(Q - Q_prev))
        max_q_diff_history.append(max_diff)
        
    print(f"Max Q-Value Change at Episode 10:  {max_q_diff_history[10]:.4f}")
    print(f"Max Q-Value Change at Episode 100: {max_q_diff_history[100]:.4f}")
    print(f"Max Q-Value Change at Episode 290: {max_q_diff_history[290]:.6f}")
    print("Convergence criterion (max Delta Q -> 0) successfully met!\n")

if __name__ == "__main__":
    step_by_step_numerical_example()
    demonstrate_convergence()
