# ============================================================
# EXPERIMENT NO : 30
# TITLE : Virtual Character Audience Engagement using Policy Gradient
# PROBLEM STATEMENT: Train virtual character to create engaging content using
# policy gradient methods for maximum audience engagement.
# DATASET : ../Datasets/Q30_Virtual_Character_PG_Dataset.csv
# ============================================================
import os, pandas as pd, numpy as np

def load_dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "../Datasets/Q30_Virtual_Character_PG_Dataset.csv"))

def get_user_inputs():
    epochs = int(input("\nEnter Policy Gradient Epochs : "))
    lr = float(input("Enter Learning Rate : "))
    return epochs, lr

def run_character_policy_gradient(dataset, epochs, lr):
    topics, scores, multipliers, n = dataset["StoryTopic"].tolist(), dataset["EngagementScore"].values, dataset["RewardMultiplier"].values, len(dataset)
    theta = np.zeros(n)

    for _ in range(epochs):
        probs = np.exp(theta) / np.sum(np.exp(theta))
        choice = np.random.choice(n, p=probs)
        grad = -probs
        grad[choice] += 1.0
        theta += lr * grad * (scores[choice] * multipliers[choice])

    final_probs = np.exp(theta) / np.sum(np.exp(theta))
    print("\n========== POLICY GRADIENT RESULT ==========")
    best_idx = np.argmax(final_probs)
    print("Optimal Content Topic  :", topics[best_idx])
    print("Selection Probability  :", round(final_probs[best_idx]*100, 2), "%")

def main():
    print("=" * 45 + "\n VIRTUAL CHARACTER POLICY GRADIENT \n" + "=" * 45)
    ds = load_dataset()
    print(ds)
    epochs, lr = get_user_inputs()
    run_character_policy_gradient(ds, epochs, lr)

if __name__ == "__main__":
    main()
