# ============================================================
# EXPERIMENT NO : 21
# TITLE : UCB Algorithm for Dynamic Streaming Content Selection
# PROBLEM STATEMENT: Use UCB algorithm to dynamically select content
# for users on a streaming platform and compare performance.
# DATASET : ../Datasets/Q21_UCB_Streaming_Dataset.csv
# ============================================================
import os, pandas as pd, numpy as np

def load_dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "../Datasets/Q21_UCB_Streaming_Dataset.csv"))

def get_user_inputs():
    rounds = int(input("\nEnter Streaming Sessions : "))
    c_param = float(input("Enter UCB Constant C (e.g. 1.5) : "))
    return rounds, c_param

def run_ucb_streaming(dataset, rounds, c_param):
    ctrs, values = dataset["BaseCTR"].values, dataset["StreamValue"].values
    n = len(ctrs)
    counts, rewards, total_val = np.zeros(n), np.zeros(n), 0.0

    for t in range(rounds):
        chosen = t if t < n else int(np.argmax((rewards / (counts + 1e-5)) + c_param * np.sqrt(np.log(t + 1) / (counts + 1e-5))))
        r = (1 if np.random.rand() < ctrs[chosen] else 0) * values[chosen]
        total_val += r
        counts[chosen] += 1
        rewards[chosen] += r

    print("\n========== UCB STREAMING RESULT ==========")
    print("Content Selection Counts :", counts.astype(int))
    print("Top Content Recommended  :", dataset["ContentName"].iloc[np.argmax(rewards)])
    print("Total Platform Value     : $", round(total_val, 2))

def main():
    print("=" * 45 + "\n UCB STREAMING CONTENT SELECTION \n" + "=" * 45)
    ds = load_dataset()
    print(ds)
    rounds, c_param = get_user_inputs()
    run_ucb_streaming(ds, rounds, c_param)

if __name__ == "__main__":
    main()
