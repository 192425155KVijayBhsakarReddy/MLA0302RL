# ============================================================
# EXPERIMENT NO : 11
# TITLE : Double DQN for Stock Trading Strategy
#
# PROBLEM STATEMENT:
# Implement Double DQN to optimize a stock trading strategy. Learn to buy,
# sell, or hold stocks to maximize profits.
#
# DATASET :
# ../Datasets/Q11_Double_DQN_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q11_Double_DQN_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== STOCK DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Training Episodes : "))
    gamma = float(input("Enter Discount Factor (Gamma) : "))
    return episodes, gamma

def run_double_dqn(dataset, episodes, gamma):
    prices = dataset["Close"].values
    n_days = len(prices)
    Q1, Q2 = np.zeros((n_days, 3)), np.zeros((n_days, 3))

    for _ in range(episodes):
        for t in range(n_days - 1):
            action = int(np.argmax(Q1[t]))
            reward = prices[t+1] - prices[t] if action == 1 else 0
            best_a = int(np.argmax(Q1[t+1]))
            target = reward + gamma * Q2[t+1, best_a]
            Q1[t, action] += 0.1 * (target - Q1[t, action])
            Q2[t] = Q1[t].copy()

    print("\n========== DOUBLE DQN RESULT ==========")
    print("Learned Q1 Table (First 3 Days):")
    print(np.round(Q1[:3], 2))

def main():
    print("=" * 45)
    print(" DOUBLE DQN STOCK TRADING AGENT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes, gamma = get_user_inputs()
    run_double_dqn(dataset, episodes, gamma)

if __name__ == "__main__":
    main()
