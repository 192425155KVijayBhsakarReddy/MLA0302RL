# ============================================================
# EXPERIMENT NO : 18
# TITLE : RL Framework for Manufacturing Process
#
# PROBLEM STATEMENT:
# Simulate an RL framework to optimize a manufacturing process, where actions
# represent machine settings and rewards are based on product quality.
#
# DATASET :
# ../Datasets/Q18_RL_Manufacturing_Process_Dataset.csv
# ============================================================

import os
import pandas as pd

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q18_RL_Manufacturing_Process_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== MANUFACTURING DATASET ==========")
    print(dataset)

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor (Gamma) : "))
    return gamma

def optimize_manufacturing(dataset, gamma):
    dataset["StateValue"] = dataset["ProductQuality"] + gamma * dataset["Reward"]
    
    print("\n========== MANUFACTURING RESULT ==========")
    print(dataset)
    
    best_setting = dataset.loc[dataset["StateValue"].idxmax()]
    print("\nOptimal Machine Setting :", best_setting["MachineSetting"])
    print("Maximum State Value     :", best_setting["StateValue"])

def main():
    print("=" * 45)
    print(" MANUFACTURING PROCESS OPTIMIZATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    gamma = get_user_inputs()
    optimize_manufacturing(dataset, gamma)

if __name__ == "__main__":
    main()
