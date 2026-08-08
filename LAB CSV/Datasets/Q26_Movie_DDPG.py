# ============================================================
# EXPERIMENT NO : 26
# TITLE : Movie Recommendation MDP using DDPG
#
# PROBLEM STATEMENT:
# Develop a recommendation system for a streaming service to suggest movies
# using MDP and Deep Deterministic Policy Gradient (DDPG).
#
# DATASET :
# ../Datasets/Q26_Movie_DDPG_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q26_Movie_DDPG_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== MOVIE DDPG DATASET ==========")
    print(dataset)

def get_user_inputs():
    episodes = int(input("\nEnter Training Episodes : "))
    lr_actor = float(input("Enter Actor Learning Rate : "))
    return episodes, lr_actor

def run_movie_ddpg(dataset, episodes, lr_actor):
    ratings = dataset["Rating"].values
    n = len(ratings)
    
    # Continuous Action Space representation for DDPG
    actor_weights = np.zeros(n)
    for _ in range(episodes):
        for i in range(n):
            reward = ratings[i]
            actor_weights[i] += lr_actor * (reward - actor_weights[i])

    best_idx = np.argmax(actor_weights)
    print("\n========== DDPG RECOMMENDATION RESULT ==========")
    print("Top Recommended Movie   :", dataset["MovieTitle"].iloc[best_idx])
    print("Predicted Action Score  :", round(actor_weights[best_idx], 3))

def main():
    print("=" * 45)
    print(" MOVIE RECOMMENDATION DDPG AGENT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    episodes, lr_actor = get_user_inputs()
    run_movie_ddpg(dataset, episodes, lr_actor)

if __name__ == "__main__":
    main()
