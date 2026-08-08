# ============================================================
# EXPERIMENT NO : 24
# TITLE : Inventory Management Using Bellman Equation
# PROBLEM STATEMENT: Use Bellman equation to find optimal ordering policy
# minimizing holding and shortage costs.
# DATASET : ../Datasets/Q24_Inventory_Bellman_Dataset.csv
# ============================================================
import os, pandas as pd, numpy as np

def load_dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "../Datasets/Q24_Inventory_Bellman_Dataset.csv"))

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor Gamma : "))
    iterations = int(input("Enter Number of Iterations : "))
    return gamma, iterations

def solve_inventory_bellman(dataset, gamma, iterations):
    levels, holding, shortage, n = dataset["StockLevel"].tolist(), dataset["HoldingCost"].values, dataset["ShortagePenalty"].values, len(dataset)
    V, order_policy = np.zeros(n), []

    for _ in range(iterations):
        for i in range(n):
            V[i] = round((holding[i] + shortage[i]) + gamma * np.mean(V), 2)

    for lvl in levels:
        order_policy.append("Reorder_Stock" if "Low" in lvl else "Hold_Current")

    print("\n========== BELLMAN INVENTORY RESULT ==========")
    for i, lvl in enumerate(levels):
        print(f"Stock Level: {lvl:<10} | Expected Cost V(s): {V[i]:<7} | Optimal Policy: {order_policy[i]}")

def main():
    print("=" * 45 + "\n INVENTORY MANAGEMENT BELLMAN OPTIMIZATION \n" + "=" * 45)
    ds = load_dataset()
    print(ds)
    gamma, iterations = get_user_inputs()
    solve_inventory_bellman(ds, gamma, iterations)

if __name__ == "__main__":
    main()
