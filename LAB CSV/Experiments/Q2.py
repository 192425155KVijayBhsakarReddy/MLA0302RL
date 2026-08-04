import pandas as pd

data=pd.read_csv("../Datasets/Q2.csv")
gamma=float(input("Gamma: "))
it=int(input("Iterations: "))
data["Value"]=data["Reward"]
for _ in range(it):
    data["Value"]=data["Reward"]+gamma*data["Value"]
print("\nPolicy Evaluation")
print(data[["StateID","StateType","Reward","Value"]])
print("Best State:",data.loc[data["Value"].idxmax(),"StateID"])
