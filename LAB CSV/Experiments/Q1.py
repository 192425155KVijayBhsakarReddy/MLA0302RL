import pandas as pd
import random

def load():
    return pd.read_csv("../Datasets/Q1.csv")

def run(data,episodes,policy):
    reward=0
    cleaned=0
    for _ in range(episodes):
        for _,r in data.iterrows():
            if r["Type"]=="Dirt":
                reward+=1
                cleaned+=1
            elif r["Type"]=="Obstacle":
                reward-=1
        if policy=="Random":
            random.shuffle(data.values)
    return reward,cleaned

data=load()
print("\n--- Cleaning Robot ---")
print(data)
ep=int(input("Episodes: "))
policy=input("Policy(Random/Greedy): ")
reward,cleaned=run(data,ep,policy)
print("\nRESULT")
print("Episodes :",ep)
print("Policy   :",policy)
print("Cleaned  :",cleaned)
print("Reward   :",reward)
