# ============================================================
# EXPERIMENT NO : 13
# TITLE : Q-Learning for Grid-Based Pac-Man Game
#
# PROBLEM STATEMENT:
# Implement Q-learning AI agent for grid Pac-Man game.
#
# DATASET :
# ../Datasets/Q13_Q_Learning_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q13_Q_Learning_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== PAC-MAN DATASET ==========")
    print(dataset)

def get_user_inputs():
    alpha = float(input("\nEnter Learning Rate (Alpha) : "))
    gamma = float(input("Enter Discount Factor (Gamma) : "))
    episodes = int(input("Enter Training Episodes : "))
    return alpha, gamma, episodes

def run_q_learning(dataset, alpha, gamma, episodes):
    states = dataset["State"].tolist()
    rewards = dataset["Reward"].values
    n_states = len(states)
    Q = np.zeros((n_states, 2))
    
    for _ in range(episodes):
        s = 0
        for _ in range(5):
            a = np.random.randint(2) if np.random.rand() < 0.1 else int(np.argmax(Q[s]))
            s_next = np.random.randint(n_states)
            r = rewards[s_next]
            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
            s = s_next

    print("\n========== Q-LEARNING RESULT ==========")
    for i, state in enumerate(states):
        print(f"State: {state:<8} | Max Q-Value: {round(np.max(Q[i]), 2)}")

def main():
    print("=" * 45)
    print(" PAC-MAN Q-LEARNING AGENT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    alpha, gamma, episodes = get_user_inputs()
    run_q_learning(dataset, alpha, gamma, episodes)

if __name__ == "__main__":
    main()
