# ============================================================
# EXPERIMENT NO : 23
# TITLE : Policy & Value Function Relationship in Gridworld
# PROBLEM STATEMENT: Demonstrate relationship between policy pi(a|s) and
# value function V(s) in a practical gridworld.
# DATASET : ../Datasets/Q23_Gridworld_Policy_Value_Dataset.csv
# ============================================================
import os, pandas as pd, numpy as np

def load_dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "../Datasets/Q23_Gridworld_Policy_Value_Dataset.csv"))

def get_user_inputs():
    gamma = float(input("\nEnter Discount Factor Gamma : "))
    iterations = int(input("Enter Evaluation Iterations : "))
    return gamma, iterations

def evaluate_policy_value(dataset, gamma, iterations):
    states, rewards, n = dataset["State"].tolist(), dataset["Reward"].values, len(dataset)
    V_rand, P_rand = np.zeros(n), np.ones((n, n)) / n
    for _ in range(iterations):
        V_rand = rewards + gamma * np.dot(P_rand, V_rand)

    V_goal, P_goal = np.zeros(n), np.full((n, n), 0.1)
    P_goal[:, 2] = 0.7
    for _ in range(iterations):
        V_goal = rewards + gamma * np.dot(P_goal, V_goal)

    print("\n========== POLICY VS VALUE FUNCTION RESULT ==========")
    for i, s in enumerate(states):
        print(f"State: {s:<4} | Random Policy V(s): {round(V_rand[i], 2):<6} | Goal Policy V(s): {round(V_goal[i], 2)}")

def main():
    print("=" * 45 + "\n POLICY AND VALUE FUNCTION RELATIONSHIP \n" + "=" * 45)
    ds = load_dataset()
    print(ds)
    gamma, iterations = get_user_inputs()
    evaluate_policy_value(ds, gamma, iterations)

if __name__ == "__main__":
    main()
