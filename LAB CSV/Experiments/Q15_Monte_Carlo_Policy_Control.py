# ============================================================
# EXPERIMENT NO : 15
# TITLE : Monte Carlo Policy Control for Call Center
#
# PROBLEM STATEMENT:
# Implement Monte Carlo Policy Control to optimize assignment of customer
# service representatives to minimize call handling time.
#
# DATASET :
# ../Datasets/Q15_Monte_Carlo_Policy_Control_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q15_Monte_Carlo_Policy_Control_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== CALL CENTER DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Number of MC Control Episodes : "))
    return episodes

def run_mc_control(dataset, episodes):
    reps = dataset["Representative"].unique()
    reward_map = dict(zip(dataset["Representative"], dataset["Reward"]))
    n_reps = len(reps)
    
    Q = np.zeros(n_reps)
    N = np.zeros(n_reps)

    for ep in range(episodes):
        arm = np.random.randint(n_reps) if np.random.rand() < 0.1 else int(np.argmax(Q))
        reward = reward_map[reps[arm]]
        N[arm] += 1
        Q[arm] += (1.0 / N[arm]) * (reward - Q[arm])

    print("\n========== MC POLICY CONTROL RESULT ==========")
    for i, r in enumerate(reps):
        print(f"Representative: {r:<5} | Action-Value Q(s,a): {round(Q[i], 2)}")
    best_rep = reps[np.argmax(Q)]
    print("\nOptimal Representative Assignment :", best_rep)

def main():
    print("=" * 45)
    print(" CALL CENTER MONTE CARLO POLICY CONTROL ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes = get_user_inputs()
    run_mc_control(dataset, episodes)

if __name__ == "__main__":
    main()
