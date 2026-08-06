# ============================================================
# EXPERIMENT NO : 17
# TITLE : OpenAI Gym MountainCar Policy Simulation
#
# PROBLEM STATEMENT:
# Implement a policy to solve the MountainCar problem by building momentum
# to reach the hill peak.
#
# DATASET :
# ../Datasets/Q17_MountainCar_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q17_MountainCar_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== MOUNTAIN CAR DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Number of Episodes : "))
    return episodes

def simulate_mountain_car(dataset, episodes):
    goal_pos = dataset["GoalPosition"].iloc[0]
    init_pos = dataset["InitialPosition"].iloc[0]
    
    successes = 0
    for _ in range(episodes):
        pos = init_pos
        vel = 0.0
        for _ in range(200):
            action = 1 if vel > 0 else -1
            vel += action * 0.001 + np.cos(3 * pos) * (-0.0025)
            pos += vel
            if pos >= goal_pos:
                successes += 1
                break

    print("\n========== MOUNTAIN CAR RESULT ==========")
    print("Total Simulation Episodes :", episodes)
    print("Successful Peak Climbs    :", successes)
    print("Success Rate              :", round((successes / episodes) * 100, 1), "%")

def main():
    print("=" * 45)
    print(" OPENAI GYM MOUNTAIN CAR SIMULATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes = get_user_inputs()
    simulate_mountain_car(dataset, episodes)

if __name__ == "__main__":
    main()
