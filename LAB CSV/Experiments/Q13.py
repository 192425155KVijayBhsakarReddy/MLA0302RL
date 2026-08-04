import pandas as pd

game_map=pd.read_csv("../Datasets/Q13.csv",header=None)

episode_count=int(input("Enter Episodes: "))
learning_rate=float(input("Enter Alpha: "))
epsilon_value=float(input("Enter Epsilon: "))

food_count=(game_map==1).sum().sum()
ghost_count=(game_map==-2).sum().sum()
wall_count=(game_map==-1).sum().sum()

agent_score=(food_count*10-ghost_count*20-wall_count*2)
final_score=agent_score*learning_rate

print("\n===== GAME MAP =====")
print(game_map)
print("\n===== Q-LEARNING RESULT =====")
print("Episodes :",episode_count)
print("Food :",food_count)
print("Ghosts :",ghost_count)
print("Walls :",wall_count)
print("Epsilon :",epsilon_value)
print("Final Score :",round(final_score,2))
