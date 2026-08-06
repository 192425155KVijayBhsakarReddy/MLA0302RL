# ============================================================
# EXPERIMENT NO : 6
# TITLE : Advertisement Bandit Optimization
#
# PROBLEM STATEMENT:
# An online platform uses bandit algorithms to decide which advertisements to
# show to users. Determine which algorithm results in the highest CTR.
#
# DATASET :
# ../Datasets/Q6_Advertisement_Bandit_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q6_Advertisement_Bandit_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== ADVERTISEMENT DATASET ==========")
    print(dataset)

def get_user_inputs():
    impressions = int(input("\nEnter Number of Impression Rounds : "))
    epsilon = float(input("Enter Epsilon Value (e.g. 0.1) : "))
    return impressions, epsilon

def run_ad_bandit(dataset, impressions, epsilon):
    ctrs = dataset["CTR"].values
    n_ads = len(ctrs)
    
    # Epsilon-Greedy Simulation
    counts = np.zeros(n_ads)
    clicks = np.zeros(n_ads)
    for t in range(impressions):
        ad = np.random.randint(n_ads) if np.random.rand() < epsilon or t < n_ads else int(np.argmax(clicks / (counts + 1e-5)))
        is_clicked = 1 if np.random.rand() < ctrs[ad] else 0
        counts[ad] += 1
        clicks[ad] += is_clicked
        
    overall_ctr = round((sum(clicks) / impressions) * 100, 2)
    print("\n========== RESULT ==========")
    print("Ad Impressions Count :", counts.astype(int))
    print("Overall CTR Achieved :", overall_ctr, "%")

def main():
    print("=" * 45)
    print(" ONLINE ADVERTISEMENT BANDIT ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    impressions, epsilon = get_user_inputs()
    run_ad_bandit(dataset, impressions, epsilon)

if __name__ == "__main__":
    main()
