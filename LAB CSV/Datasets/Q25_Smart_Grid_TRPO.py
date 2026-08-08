# ============================================================
# EXPERIMENT NO : 25
# TITLE : Smart Grid Energy Management using TRPO
#
# PROBLEM STATEMENT:
# Model a smart grid managing energy consumption and production to balance
# supply/demand and minimize cost using Trust Region Policy Optimization (TRPO).
#
# DATASET :
# ../Datasets/Q25_Smart_Grid_TRPO_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q25_Smart_Grid_TRPO_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== SMART GRID DATASET ==========")
    print(dataset)

def get_user_inputs():
    kl_bound = float(input("\nEnter TRPO KL-Divergence Bound (e.g. 0.01) : "))
    epochs = int(input("Enter Optimization Epochs : "))
    return kl_bound, epochs

def run_trpo_smart_grid(dataset, kl_bound, epochs):
    demand = dataset["EnergyDemand"].values
    capacity = dataset["ProductionCapacity"].values
    cost = dataset["GridCost"].values
    n = len(demand)

    policy_weights = np.zeros(n)
    for epoch in range(epochs):
        for i in range(n):
            deficit = max(0, demand[i] - capacity[i])
            reward = -(deficit * cost[i])
            # TRPO constrained update simulation step
            step_size = min(kl_bound, 0.05)
            policy_weights[i] += step_size * reward * 0.01

    print("\n========== TRPO SMART GRID RESULT ==========")
    for i, slot in enumerate(dataset["TimeSlot"]):
        dispatch = min(capacity[i], demand[i] + policy_weights[i])
        print(f"Slot: {slot:<15} | Demand: {demand[i]:<4} | Optimized Dispatch: {round(dispatch, 1)}")

def main():
    print("=" * 45)
    print(" SMART GRID TRPO ENERGY MANAGEMENT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    kl_bound, epochs = get_user_inputs()
    run_trpo_smart_grid(dataset, kl_bound, epochs)

if __name__ == "__main__":
    main()
