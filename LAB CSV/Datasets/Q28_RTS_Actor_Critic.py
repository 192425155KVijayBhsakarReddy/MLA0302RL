# ============================================================
# EXPERIMENT NO : 28
# TITLE : RTS Game AI Agent using Actor-Critic
#
# PROBLEM STATEMENT:
# Train an Actor-Critic AI agent to play a real-time strategy game (RTS) by
# optimizing resource allocation and combat strategy.
#
# DATASET :
# ../Datasets/Q28_RTS_Actor_Critic_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q28_RTS_Actor_Critic_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== RTS GAME DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Training Episodes : "))
    gamma = float(input("Enter Discount Factor (Gamma) : "))
    return episodes, gamma

def run_rts_actor_critic(dataset, episodes, gamma):
    states = dataset["State"].tolist()
    actions = dataset["ActionStrategy"].tolist()
    metal = dataset["ResourceMetal"].values
    n = len(states)

    V = np.zeros(n)
    policy_scores = np.zeros(n)

    for _ in range(episodes):
        for i in range(n):
            reward = metal[i] / 100.0
            advantage = reward + gamma * np.mean(V) - V[i]
            V[i] += 0.1 * advantage
            policy_scores[i] += 0.05 * advantage

    print("\n========== RTS ACTOR-CRITIC RESULT ==========")
    for i, st in enumerate(states):
        print(f"State: {st:<12} | Optimal Action: {actions[i]:<18} | Value: {round(V[i], 2)}")

def main():
    print("=" * 45)
    print(" RTS GAME ACTOR-CRITIC AGENT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes, gamma = get_user_inputs()
    run_rts_actor_critic(dataset, episodes, gamma)

if __name__ == "__main__":
    main()
