# ============================================================
# EXPERIMENT NO : 4
# TITLE : Policy Iteration for Delivery Drone
#
# PROBLEM STATEMENT:
# A delivery drone needs to find the shortest path in a city grid.
# Implement policy iteration algorithm using dynamic programming in Python.
#
# DATASET :
# ../Datasets/Q4_Policy_Iteration_Dataset.csv
# ============================================================

import os
import pandas as pd

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q4_Policy_Iteration_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== DRONE DATASET ==========")
    print(dataset)

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor (Gamma) : "))
    iterations = int(input("Enter Evaluation Iterations : "))
    return gamma, iterations

def perform_policy_iteration(dataset, gamma, iterations):
    locations = dataset["Location"].tolist()
    rewards = dict(zip(dataset["Location"], dataset["Reward"]))
    values = {loc: 0.0 for loc in locations}
    policy = {loc: "Move_Forward" for loc in locations}

    for _ in range(iterations):
        for loc in locations:
            reward = rewards[loc]
            avg_val = sum(values[l] for l in locations) / len(locations)
            values[loc] = round(reward + gamma * avg_val, 2)

    print("\n========== POLICY ITERATION RESULT ==========")
    for loc in locations:
        print(f"Location: {loc:<10} | Value: {values[loc]:<6} | Policy: {policy[loc]}")

def main():
    print("=" * 45)
    print(" DELIVERY DRONE POLICY ITERATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    gamma, iterations = get_user_inputs()
    perform_policy_iteration(dataset, gamma, iterations)

if __name__ == "__main__":
    main()
