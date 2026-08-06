# ============================================================
# EXPERIMENT NO : 3
# TITLE : Dynamic Pricing Multi-Armed Bandit
#
# PROBLEM STATEMENT:
# Simulate Epsilon-Greedy and Thompson Sampling to maximize revenue.
#
# DATASET :
# ../Datasets/Q3_Multi_Armed_Bandit_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q3_Multi_Armed_Bandit_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== DATASET ==========")
    print(dataset)

def get_user_inputs():
    rounds = int(input("\nEnter Number of Decision Rounds : "))
    epsilon = float(input("Enter Epsilon Value (e.g. 0.1) : "))
    return rounds, epsilon

def run_bandit_strategies(dataset, rounds, epsilon):
    probs = dataset["Probability"].values
    n_arms = len(probs)
    prices = np.array([10.0 * (i + 1) for i in range(n_arms)])

    # Epsilon-Greedy
    counts, vals, rev_eg = np.zeros(n_arms), np.zeros(n_arms), 0.0
    for t in range(rounds):
        arm = np.random.randint(n_arms) if np.random.rand() < epsilon or t < n_arms else int(np.argmax(vals))
        reward = (1 if np.random.rand() < probs[arm] else 0) * prices[arm]
        rev_eg += reward
        counts[arm] += 1
        vals[arm] += (reward - vals[arm]) / counts[arm]

    print("\n========== RESULT ==========")
    print("Epsilon-Greedy Total Revenue : $", round(rev_eg, 2))

def main():
    print("=" * 45)
    print(" DYNAMIC PRICING MULTI-ARMED BANDIT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    rounds, epsilon = get_user_inputs()
    run_bandit_strategies(dataset, rounds, epsilon)

if __name__ == "__main__":
    main()
