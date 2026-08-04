import pandas as pd

def load_dataset():
    return pd.read_csv("../Datasets/Q16.csv")

def initialize_values(dataset):
    dataset["StateValue"] = dataset["Reward"]
    return dataset

def bellman_update(dataset, gamma, iterations):
    for current_iteration in range(iterations):
        updated_values = []
        for _, current_state in dataset.iterrows():
            reward = current_state["Reward"]
            previous_value = current_state["StateValue"]
            new_value = reward + gamma * previous_value
            updated_values.append(round(new_value, 2))
        dataset["StateValue"] = updated_values
    return dataset

def display_results(dataset):
    print("\n========== FINAL STATE VALUES ==========")
    print(dataset)

    best_state = dataset.loc[dataset["StateValue"].idxmax()]
    print("\nOptimal State :", best_state["State"])
    print("Maximum Value :", best_state["StateValue"])

    print("\nOptimal Path")
    ordered = dataset.sort_values(by="StateValue", ascending=False)
    print(" -> ".join(ordered["State"]) + " -> END")

def main():
    print("=" * 45)
    print(" BELLMAN OPTIMALITY - ROBOT NAVIGATION ")
    print("=" * 45)

    robot_dataset = load_dataset()

    print("\nDataset")
    print(robot_dataset)

    gamma = float(input("\nEnter Discount Factor (Gamma): "))
    iterations = int(input("Enter Number of Iterations: "))

    robot_dataset = initialize_values(robot_dataset)
    result_dataset = bellman_update(robot_dataset, gamma, iterations)

    display_results(result_dataset)

if __name__ == "__main__":
    main()
