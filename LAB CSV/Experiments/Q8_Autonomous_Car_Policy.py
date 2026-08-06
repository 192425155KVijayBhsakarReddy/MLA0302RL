# ============================================================
# EXPERIMENT NO : 8
# TITLE : Autonomous Car Road Network Policy
#
# PROBLEM STATEMENT:
# Simulate an autonomous car navigating a road network with intersections.
# Design policies to follow traffic rules and evaluate effectiveness.
#
# DATASET :
# ../Datasets/Q8_Autonomous_Car_Policy_Dataset.csv
# ============================================================

import os
import pandas as pd

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q8_Autonomous_Car_Policy_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== ROAD NETWORK DATASET ==========")
    print(dataset)

def get_user_inputs():
    red_signal_penalty = float(input("\nEnter Waiting Penalty for Red Signal : "))
    return red_signal_penalty

def evaluate_car_policy(dataset, red_signal_penalty):
    total_distance = 0
    total_time = 0.0

    for _, row in dataset.iterrows():
        dist = row["Distance"]
        signal = str(row["Signal"]).strip().lower()
        total_distance += dist
        if signal == "red":
            total_time += dist + red_signal_penalty
        else:
            total_time += dist

    print("\n========== POLICY RESULT ==========")
    print("Total Distance Traveled :", total_distance, "km")
    print("Total Estimated Time    :", round(total_time, 2), "mins")

def main():
    print("=" * 45)
    print(" AUTONOMOUS CAR ROAD NETWORK POLICY ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    red_signal_penalty = get_user_inputs()
    evaluate_car_policy(dataset, red_signal_penalty)

if __name__ == "__main__":
    main()
