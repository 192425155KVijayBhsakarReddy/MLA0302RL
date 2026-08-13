# ============================================================
# EXPERIMENT NO : 32
# TITLE : Retail Inventory Management using Model-Based RL
#
# PROBLEM STATEMENT:
# Develop a data generation model simulating customer demand patterns and evaluate
# retail inventory management policies using Model-Based RL.
#
# DATASET :
# ../Datasets/Q32_Retail_Inventory_MBRL_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q32_Retail_Inventory_MBRL_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== RETAIL INVENTORY DATASET ==========")
    print(dataset)

def get_user_inputs():
    days = int(input("\nEnter Simulation Horizon Days : "))
    reorder_point = int(input("Enter Reorder Threshold Level : "))
    return days, reorder_point

def simulate_retail_mbrl(dataset, days, reorder_point):
    products = dataset["ProductID"].tolist()
    demands = dataset["DailyDemandMean"].values
    costs = dataset["UnitCost"].values

    total_cost = 0.0
    for i, prod in enumerate(products):
        stock = reorder_point * 2
        for _ in range(days):
            sim_demand = np.random.poisson(demands[i])
            stock -= sim_demand
            if stock <= reorder_point:
                stock += reorder_point * 2
                total_cost += costs[i] * 10

    print("\n========== MODEL-BASED RL RESULT ==========")
    print("Total Simulated Holding & Order Cost: $", round(total_cost, 2))
    print("Recommended Inventory Policy       : Maintain Reorder Point at", reorder_point)

def main():
    print("=" * 45)
    print(" RETAIL INVENTORY MODEL-BASED RL ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    days, reorder_point = get_user_inputs()
    simulate_retail_mbrl(dataset, days, reorder_point)

if __name__ == "__main__":
    main()
