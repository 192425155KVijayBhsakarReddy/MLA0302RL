# ============================================================
# EXPERIMENT NO : 19
# TITLE : Monte Carlo Evaluation for Customer Churn
#
# PROBLEM STATEMENT:
# Use Monte Carlo methods to evaluate a policy for predicting customer churn
# in a subscription-based service.
#
# DATASET :
# ../Datasets/Q19_Monte_Carlo_Customer_Churn_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q19_Monte_Carlo_Customer_Churn_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== CUSTOMER CHURN DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Number of Monte Carlo Episodes : "))
    return episodes

def run_churn_evaluation(dataset, episodes):
    customers = dataset["CustomerID"].tolist()
    rewards = dataset["Reward"].values
    
    returns = {c: [] for c in customers}
    for _ in range(episodes):
        for i, c in enumerate(customers):
            g = rewards[i] * (1.2 if dataset["ChurnStatus"].iloc[i] == "No" else 0.5)
            returns[c].append(g)

    print("\n========== MONTE CARLO RESULT ==========")
    for c in customers:
        avg_val = round(np.mean(returns[c]), 2)
        print(f"Customer ID: {c:<5} | Expected Value V(s): {avg_val}")

def main():
    print("=" * 45)
    print(" CUSTOMER CHURN MONTE CARLO EVALUATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes = get_user_inputs()
    run_churn_evaluation(dataset, episodes)

if __name__ == "__main__":
    main()
