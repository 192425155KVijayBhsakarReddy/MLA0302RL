import pandas as pd
d=pd.read_csv("../Datasets/Q5.csv")
gamma=float(input("Gamma: "))
it=int(input("Iterations: "))
d["Value"]=d["Reward"]
for _ in range(it):
    d["Value"]=d["Reward"]+gamma*d["Value"]
print("\nValue Iteration")
print(d)
print("Optimal Location:",
      d.loc[d["Value"].idxmax(),"LocationID"])
