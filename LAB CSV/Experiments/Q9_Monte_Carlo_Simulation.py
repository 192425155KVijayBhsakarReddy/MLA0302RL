# ============================================================
# EXPERIMENT NO : 9
# TITLE : Monte Carlo Simulation for Call Center Assignment
#
# PROBLEM STATEMENT:
# A call center wants to optimize the assignment of representatives. Implement
# Monte Carlo simulation to estimate value function for assignment policies.
#
# DATASET :
# ../Datasets/Q9_Monte_Carlo_Simulation_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q9_Monte_Carlo_Simulation_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== CALL CENTER DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Number of Monte Carlo Episodes : "))
    return episodes

def run_monte_carlo_simulation(dataset, episodes):
    reps = dataset["Representative"].unique()
    reward_map = dict(zip(dataset["Representative"], dataset["Reward"]))
    
    returns = {r: [] for r in reps}
    for _ in range(episodes):
        sampled_rep = np.random.choice(reps)
        reward = reward_map[sampled_rep]
        returns[sampled_rep].append(reward)

    print("\n========== MONTE CARLO RESULT ==========")
    for r in reps:
        avg_val = round(np.mean(returns[r]), 2) if returns[r] else 0
        print(f"Representative: {r:<5} | Estimated Value V(s): {avg_val}")

def main():
    print("=" * 45)
    print(" CALL CENTER MONTE CARLO SIMULATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes = get_user_inputs()
    run_monte_carlo_simulation(dataset, episodes)

if __name__ == "__main__":
    main()
