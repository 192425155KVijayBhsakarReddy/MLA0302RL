# ============================================================
# EXPERIMENT NO : 22
# TITLE : K-Armed Bandit for Marketing Campaign Choices
# PROBLEM STATEMENT: Simulate k-armed bandit to optimize marketing choices
# using epsilon-greedy, UCB, and Thompson Sampling algorithms.
# DATASET : ../Datasets/Q22_Marketing_Bandit_Dataset.csv
# ============================================================
import os, pandas as pd, numpy as np

def load_dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "../Datasets/Q22_Marketing_Bandit_Dataset.csv"))

def get_user_inputs():
    rounds = int(input("\nEnter Marketing Rounds : "))
    epsilon = float(input("Enter Epsilon Exploration Rate : "))
    return rounds, epsilon

def run_marketing_bandit(dataset, rounds, epsilon):
    rates, n = dataset["ConversionRate"].values, len(dataset)
    counts_eg, vals_eg, rev_eg = np.zeros(n), np.zeros(n), 0.0
    for t in range(rounds):
        arm = np.random.randint(n) if np.random.rand() < epsilon or t < n else int(np.argmax(vals_eg))
        r = 1 if np.random.rand() < rates[arm] else 0
        rev_eg += r
        counts_eg[arm] += 1
        vals_eg[arm] += (r - vals_eg[arm]) / counts_eg[arm]

    succ, fail, rev_ts = np.ones(n), np.ones(n), 0.0
    for t in range(rounds):
        arm = int(np.argmax(np.random.beta(succ, fail)))
        r = 1 if np.random.rand() < rates[arm] else 0
        rev_ts += r
        if r: succ[arm] += 1
        else: fail[arm] += 1

    print("\n========== MARKETING BANDIT RESULT ==========")
    print("Epsilon-Greedy Conversions :", int(rev_eg))
    print("Thompson Sampling Conversions:", int(rev_ts))

def main():
    print("=" * 45 + "\n MARKETING CAMPAIGN K-ARMED BANDIT \n" + "=" * 45)
    ds = load_dataset()
    print(ds)
    rounds, epsilon = get_user_inputs()
    run_marketing_bandit(ds, rounds, epsilon)

if __name__ == "__main__":
    main()
