# ============================================================
# EXPERIMENT NO : 10
# TITLE : Policy Gradient Method for Financial Investment
#
# PROBLEM STATEMENT:
# Optimize investment strategy using basic policy gradient method to simulate
# and optimize policy for maximum returns.
#
# DATASET :
# ../Datasets/Q10_Policy_Gradient_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q10_Policy_Gradient_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== INVESTMENT DATASET ==========")
    print(dataset)

def get_user_inputs():
    lr = float(input("\nEnter Learning Rate (Alpha) : "))
    epochs = int(input("Enter Training Epochs : "))
    return lr, epochs

def run_policy_gradient(dataset, lr, epochs):
    returns = dataset["Return"].values
    n_actions = 3
    theta = np.zeros(n_actions)

    for _ in range(epochs):
        probs = np.exp(theta) / np.sum(np.exp(theta))
        action = np.random.choice(n_actions, p=probs)
        reward = np.random.choice(returns) * (action + 1)
        
        gradient = -probs
        gradient[action] += 1.0
        theta += lr * gradient * reward

    final_probs = np.round(np.exp(theta) / np.sum(np.exp(theta)), 3)
    print("\n========== POLICY GRADIENT RESULT ==========")
    print("Action Preference Weights Theta :", np.round(theta, 3))
    print("Optimal Strategy Probabilities :", final_probs)

def main():
    print("=" * 45)
    print(" FINANCIAL POLICY GRADIENT OPTIMIZATION ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    lr, epochs = get_user_inputs()
    run_policy_gradient(dataset, lr, epochs)

if __name__ == "__main__":
    main()
