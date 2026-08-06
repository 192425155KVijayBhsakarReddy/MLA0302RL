# ============================================================
# EXPERIMENT NO : 20
# TITLE : Epsilon-Greedy Content Recommendation
#
# PROBLEM STATEMENT:
# Implement an epsilon-greedy strategy to optimize content recommendations on
# an online learning platform.
#
# DATASET :
# ../Datasets/Q20_Epsilon_Greedy_Content_Recommendation_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q20_Epsilon_Greedy_Content_Recommendation_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== CONTENT RECOMMENDATION DATASET ==========")
    print(dataset)

def get_user_inputs():
    runs = int(input("\nEnter Number of Recommendation Runs : "))
    epsilon = float(input("Enter Epsilon Exploration Rate : "))
    return runs, epsilon

def run_epsilon_greedy(dataset, runs, epsilon):
    items = dataset["ContentID"].tolist()
    click_probs = dataset["ClickProbability"].values
    rewards = dataset["Reward"].values
    n_items = len(items)

    counts = np.zeros(n_items)
    estimated_vals = np.zeros(n_items)

    for t in range(runs):
        chosen = np.random.randint(n_items) if np.random.rand() < epsilon or t < n_items else int(np.argmax(estimated_vals))
        r = (1 if np.random.rand() < click_probs[chosen] else 0) * rewards[chosen]
        counts[chosen] += 1
        estimated_vals[chosen] += (r - estimated_vals[chosen]) / counts[chosen]

    print("\n========== RESULT ==========")
    print("Content Selection Counts :", counts.astype(int))
    best_item = items[np.argmax(estimated_vals)]
    print("Optimal Content Choice   :", best_item)

def main():
    print("=" * 45)
    print(" EPSILON-GREEDY CONTENT RECOMMENDATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    runs, epsilon = get_user_inputs()
    run_epsilon_greedy(dataset, runs, epsilon)

if __name__ == "__main__":
    main()
