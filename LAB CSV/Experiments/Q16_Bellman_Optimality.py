# ============================================================
# EXPERIMENT NO : 16
# TITLE : Bellman's Optimality Equation for Robot Navigation
#
# PROBLEM STATEMENT:
# Compute the optimal state-value function V*(s) for robot navigation tasks
# using Bellman's optimality equation.
#
# DATASET :
# ../Datasets/Q16_Bellman_Optimality_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q16_Bellman_Optimality_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== DATASET ==========")
    print(dataset)

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor (Gamma) : "))
    iterations = int(input("Enter Number of Iterations : "))
    return gamma, iterations

def solve_bellman_optimality(dataset, gamma, iterations):
    states = dataset["State"].tolist()
    rewards = dataset["Reward"].values
    n = len(states)
    V = np.zeros(n)

    for _ in range(iterations):
        for i in range(n):
            reward = rewards[i]
            avg_next = sum(V) / n
            V[i] = round(reward + gamma * avg_next, 2)

    print("\n========== BELLMAN OPTIMALITY RESULT ==========")
    for i, s in enumerate(states):
        print(f"State: {s:<5} | Optimal State Value V*(s): {V[i]}")

def main():
    print("=" * 45)
    print(" BELLMAN OPTIMALITY EQUATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    gamma, iterations = get_user_inputs()
    solve_bellman_optimality(dataset, gamma, iterations)

if __name__ == "__main__":
    main()
