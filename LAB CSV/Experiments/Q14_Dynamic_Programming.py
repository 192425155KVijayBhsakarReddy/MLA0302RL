# ============================================================
# EXPERIMENT NO : 14
# TITLE : Dynamic Programming Gridworld Navigation
#
# PROBLEM STATEMENT:
# Implement dynamic programming methods to solve a gridworld navigation
# problem reaching the goal with least steps while avoiding obstacles.
#
# DATASET :
# ../Datasets/Q14_Dynamic_Programming_Dataset.csv
# ============================================================

import os
import pandas as pd

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q14_Dynamic_Programming_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== GRIDWORLD DATASET ==========")
    print(dataset)

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor (Gamma) : "))
    iterations = int(input("Enter Number of Iterations : "))
    return gamma, iterations

def perform_dp(dataset, gamma, iterations):
    cells = dataset["Cell"].tolist()
    rewards = dict(zip(dataset["Cell"], dataset["Reward"]))
    values = {c: 0.0 for c in cells}

    for _ in range(iterations):
        for c in cells:
            reward = rewards[c]
            avg_next = sum(values[cx] for cx in cells) / len(cells)
            values[c] = round(reward + gamma * avg_next, 2)

    print("\n========== DYNAMIC PROGRAMMING RESULT ==========")
    for c in cells:
        print(f"Cell: {c:<5} | Optimal Value V*(s): {values[c]}")

def main():
    print("=" * 45)
    print(" GRIDWORLD DYNAMIC PROGRAMMING ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    gamma, iterations = get_user_inputs()
    perform_dp(dataset, gamma, iterations)

if __name__ == "__main__":
    main()
