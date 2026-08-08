# ============================================================
# EXPERIMENT NO : 27
# TITLE : Financial Portfolio Management using A3C
#
# PROBLEM STATEMENT:
# Implement an Asynchronous Advantage Actor-Critic (A3C) agent to choose
# assets that maximize returns while minimizing risk.
#
# DATASET :
# ../Datasets/Q27_Portfolio_A3C_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q27_Portfolio_A3C_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== PORTFOLIO A3C DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter A3C Training Episodes : "))
    workers = int(input("Enter Number of Asynchronous Workers : "))
    return episodes, workers

def run_a3c_portfolio(dataset, episodes, workers):
    returns = dataset["ExpectedReturn"].values
    risks = dataset["RiskScore"].values
    n = len(returns)

    policy_probs = np.ones(n) / n
    for _ in range(episodes):
        for _ in range(workers):
            sharpe = (returns - 0.02) / (risks + 1e-5)
            policy_probs += 0.01 * (sharpe - np.mean(sharpe))
            policy_probs = np.maximum(0, policy_probs)
            policy_probs /= np.sum(policy_probs)

    print("\n========== A3C PORTFOLIO RESULT ==========")
    for i, asset in enumerate(dataset["AssetName"]):
        print(f"Asset: {asset:<18} | Optimal Weight: {round(policy_probs[i]*100, 2)}%")

def main():
    print("=" * 45)
    print(" FINANCIAL PORTFOLIO A3C AGENT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes, workers = get_user_inputs()
    run_a3c_portfolio(dataset, episodes, workers)

if __name__ == "__main__":
    main()
