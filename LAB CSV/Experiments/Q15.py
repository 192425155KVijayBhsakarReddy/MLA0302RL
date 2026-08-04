import pandas as pd

call_center_data=pd.read_csv("../Datasets/Q15.csv")

episode_count=int(input("Enter Episodes: "))
discount_factor=float(input("Enter Gamma: "))

average_reward=call_center_data["Reward"].mean()*discount_factor

for episode in range(episode_count-1):
    current_reward=call_center_data["Reward"].mean()*discount_factor
    average_reward=(average_reward+current_reward)/2

best_representative=call_center_data.groupby("Representative")["Reward"].mean().idxmax()

print("\n===== CALL CENTER DATA =====")
print(call_center_data.head())
print("\n===== MONTE CARLO RESULT =====")
print("Average Reward :",round(average_reward,2))
print("Best Representative :",best_representative)
