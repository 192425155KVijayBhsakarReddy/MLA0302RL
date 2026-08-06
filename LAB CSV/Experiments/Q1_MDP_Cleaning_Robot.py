# ============================================================
# EXPERIMENT NO : 1
# TITLE : MDP Based Autonomous Cleaning Robot
#
# PROBLEM STATEMENT:
# An autonomous cleaning robot navigates a grid where certain cells contain
# dirt (reward:+1) and obstacles (penalty:-1). Find optimal policy using MDP.
#
# DATASET :
# ../Datasets/Q1_MDP_Cleaning_Robot_Dataset.csv
# ============================================================

import os
import pandas as pd
import random

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q1_MDP_Cleaning_Robot_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Number of Episodes : "))
    dirt_reward = float(input("Enter Reward for Dirt Cell : "))
    obstacle_penalty = float(input("Enter Penalty for Obstacle Cell : "))
    return episodes, dirt_reward, obstacle_penalty

def simulate_robot(dataset, episodes, dirt_reward, obstacle_penalty):
    cleaned, obstacles, total_reward = 0, 0, 0.0
    for _ in range(episodes):
        for _, row in dataset.iterrows():
            ctype = str(row["Type"]).lower()
            if ctype == "dirt":
                cleaned += 1
                total_reward += dirt_reward
            elif ctype == "obstacle":
                obstacles += 1
                total_reward += obstacle_penalty
            elif ctype == "goal":
                total_reward += float(row["Reward"])
    
    print("\n========== RESULT ==========")
    print("Cleaned Cells    :", cleaned)
    print("Obstacle Hits    :", obstacles)
    print("Total Reward     :", round(total_reward, 2))

def main():
    print("=" * 45)
    print(" MDP BASED AUTONOMOUS CLEANING ROBOT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes, dirt_reward, obstacle_penalty = get_user_inputs()
    simulate_robot(dataset, episodes, dirt_reward, obstacle_penalty)

if __name__ == "__main__":
    main()
