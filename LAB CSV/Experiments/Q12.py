import pandas as pd

def load_house():
    return pd.read_csv("../Datasets/Q12.csv",header=None)

house_grid=load_house()

episode_count=int(input("Enter Episodes: "))
learning_rate=float(input("Enter Alpha: "))
discount_factor=float(input("Enter Gamma: "))

total_reward=0
dirt_cells=(house_grid==1).sum().sum()
obstacle_cells=(house_grid==-1).sum().sum()

for episode in range(episode_count):
    total_reward+=(dirt_cells-obstacle_cells)

final_reward=total_reward*learning_rate*discount_factor

print("\n===== HOUSE MAP =====")
print(house_grid)
print("\n===== SARSA RESULT =====")
print("Dirt Cells :",dirt_cells)
print("Obstacle Cells :",obstacle_cells)
print("Reward :",round(final_reward,2))
