# ============================================================
# EXPERIMENT NO : 29
# TITLE : Autonomous Vehicle Racing Driving Policy using A2C
#
# PROBLEM STATEMENT:
# Train an AI agent to compete in autonomous vehicle racing competitions using
# Advantage Actor-Critic (A2C) to optimize lap times and speed.
#
# DATASET :
# ../Datasets/Q29_Autonomous_Racing_A2C_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q29_Autonomous_Racing_A2C_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== RACING TRACK DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Training Laps : "))
    lr_a2c = float(input("Enter A2C Learning Rate : "))
    return episodes, lr_a2c

def run_racing_a2c(dataset, episodes, lr_a2c):
    segments = dataset["TrackSegment"].tolist()
    opt_speeds = dataset["OptimalSpeed"].values
    penalties = dataset["LapTimePenalty"].values
    n = len(segments)

    learned_speeds = np.full(n, 80.0)
    for _ in range(episodes):
        for i in range(n):
            advantage = opt_speeds[i] - learned_speeds[i] - penalties[i]
            learned_speeds[i] += lr_a2c * advantage

    print("\n========== A2C RACING RESULT ==========")
    for i, seg in enumerate(segments):
        print(f"Segment: {seg:<15} | Optimized Target Speed: {round(learned_speeds[i], 1)} km/h")

def main():
    print("=" * 45)
    print(" AUTONOMOUS RACING A2C DRIVING AGENT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes, lr_a2c = get_user_inputs()
    run_racing_a2c(dataset, episodes, lr_a2c)

if __name__ == "__main__":
    main()
