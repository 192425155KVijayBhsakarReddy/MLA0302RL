# ============================================================
# EXPERIMENT NO : 12
# TITLE : SARSA Algorithm for Vacuum Cleaner Robot
#
# PROBLEM STATEMENT:
# Use SARSA algorithm to learn optimal vacuum cleaning policy.
#
# DATASET :
# ../Datasets/Q12_SARSA_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q12_SARSA_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== VACUUM DATASET ==========")
    print(dataset)

def get_user_inputs():
    alpha = float(input("\nEnter Learning Rate (Alpha) : "))
    gamma = float(input("Enter Discount Factor (Gamma) : "))
    episodes = int(input("Enter Training Episodes : "))
    return alpha, gamma, episodes

def run_sarsa(dataset, alpha, gamma, episodes):
    rooms = dataset["Room"].tolist()
    rewards = dataset["Reward"].values
    n_states = len(rooms)
    Q = np.zeros((n_states, 2))
    
    for _ in range(episodes):
        s = np.random.randint(n_states)
        a = int(np.argmax(Q[s]))
        for _ in range(3):
            r = rewards[s]
            s_next = (s + 1) % n_states
            a_next = int(np.argmax(Q[s_next]))
            Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
            s, a = s_next, a_next

    print("\n========== SARSA RESULT ==========")
    for i, room in enumerate(rooms):
        print(f"Room: {room:<5} | Max Q-Value: {round(np.max(Q[i]), 2)}")

def main():
    print("=" * 45)
    print(" VACUUM CLEANER SARSA ALGORITHM ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    alpha, gamma, episodes = get_user_inputs()
    run_sarsa(dataset, alpha, gamma, episodes)

if __name__ == "__main__":
    main()
