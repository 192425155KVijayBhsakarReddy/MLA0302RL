# ============================================================
# EXPERIMENT NO : 2
# TITLE : Policy Evaluation for Warehouse Robot
#
# PROBLEM STATEMENT:
# A robot navigates a warehouse to pick and place items. Implement a policy
# evaluation algorithm to determine the value function for a given policy.
#
# DATASET :
# ../Datasets/Q2_Policy_Evaluation_Dataset.csv
# ============================================================

import os
import pandas as pd

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q2_Policy_Evaluation_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== WAREHOUSE DATASET ==========")
    print(dataset)

def get_user_inputs():
    discount_factor = float(input("\nEnter Discount Factor (Gamma) : "))
    iterations = int(input("Enter Number of Iterations : "))
    return discount_factor, iterations

def perform_policy_evaluation(dataset, discount_factor, iterations):
    dataset["StateValue"] = dataset["Reward"].astype(float)
    
    for _ in range(iterations):
        new_values = []
        for _, row in dataset.iterrows():
            reward = row["Reward"]
            prev_val = row["StateValue"]
            new_val = reward + discount_factor * prev_val
            new_values.append(round(new_val, 2))
        dataset["StateValue"] = new_values

    print("\n========== POLICY EVALUATION RESULT ==========")
    print(dataset)
    
    best_row = dataset.loc[dataset["StateValue"].idxmax()]
    print("\nBest State          :", best_row["State"])
    print("Maximum State Value :", best_row["StateValue"])

def main():
    print("=" * 45)
    print(" WAREHOUSE ROBOT POLICY EVALUATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    discount_factor, iterations = get_user_inputs()
    perform_policy_evaluation(dataset, discount_factor, iterations)

if __name__ == "__main__":
    main()