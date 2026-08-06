# ============================================================
# EXPERIMENT NO : 5
# TITLE : Value Iteration for Taxi Dispatching System
#
# PROBLEM STATEMENT:
# In a taxi dispatching system, use value iteration to find the optimal
# dispatch policy to reach pick-up points quickly.
#
# DATASET :
# ../Datasets/Q5_Value_Iteration_Dataset.csv
# ============================================================

import os
import pandas as pd

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q5_Value_Iteration_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== TAXI DATASET ==========")
    print(dataset)

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor (Gamma) : "))
    iterations = int(input("Enter Number of Iterations : "))
    return gamma, iterations

def perform_value_iteration(dataset, gamma, iterations):
    locations = dataset["LocationID"].tolist()
    rewards = dict(zip(dataset["LocationID"], dataset["Reward"]))
    values = {loc: 0.0 for loc in locations}
    policy = {}

    for _ in range(iterations):
        for loc in locations:
            reward = rewards[loc]
            avg_next = sum(values[l] for l in locations) / len(locations)
            values[loc] = round(reward + gamma * avg_next, 2)
            policy[loc] = "Dispatch_Taxi" if reward >= 0 else "Wait"

    print("\n========== VALUE ITERATION RESULT ==========")
    for loc in locations:
        print(f"Location: {loc:<6} | Value: {values[loc]:<6} | Policy: {policy[loc]}")

def main():
    print("=" * 45)
    print(" TAXI DISPATCH VALUE ITERATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    gamma, iterations = get_user_inputs()
    perform_value_iteration(dataset, gamma, iterations)

if __name__ == "__main__":
    main()
