# ============================================================
# EXPERIMENT NO : 7
# TITLE : State-Value Computation Using Bellman Equation
#
# PROBLEM STATEMENT:
# A delivery robot operates in a warehouse with predefined delivery points.
# Compute the state-value function using Bellman equations.
#
# DATASET :
# ../Datasets/Q7_Bellman_Equation_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q7_Bellman_Equation_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== DATASET ==========")
    print(dataset)

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor (Gamma) : "))
    iterations = int(input("Enter Number of Iterations : "))
    return gamma, iterations

def compute_bellman(dataset, gamma, iterations):
    rewards = dataset["Reward"].values.astype(float)
    n_states = len(rewards)
    state_values = np.zeros(n_states)
    
    # Transition matrix (equal probability transition)
    P = np.ones((n_states, n_states)) / n_states

    for _ in range(iterations):
        state_values = rewards + gamma * np.dot(P, state_values)

    dataset["StateValue"] = np.round(state_values, 2)
    print("\n========== BELLMAN RESULT ==========")
    print(dataset)

def main():
    print("=" * 45)
    print(" BELLMAN EQUATION STATE-VALUE COMPUTATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    gamma, iterations = get_user_inputs()
    compute_bellman(dataset, gamma, iterations)

if __name__ == "__main__":
    main()
