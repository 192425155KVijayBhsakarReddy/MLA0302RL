import pandas as pd,random,math
d=pd.read_csv("../Datasets/Q3.csv")
algo=input("Algorithm(Epsilon/UCB/Thompson): ")
rounds=int(input("Rounds: "))
eps=float(input("Epsilon: "))
Q=[0.0]*len(d);N=[0]*len(d)
for t in range(rounds):
    if algo.lower()=="ucb" and t>=len(d):
        a=max(range(len(d)),key=lambda i:Q[i]+math.sqrt(2*math.log(t+1)/(N[i]+1))
    elif algo.lower()=="epsilon":
        a=random.randrange(len(d)) if random.random()<eps else Q.index(max(Q))
    else:
        a=random.randrange(len(d))
    r=1 if random.random()<d.loc[a,"PurchaseProbability"] else 0
    N[a]+=1
    Q[a]+= (r-Q[a])/N[a]
print(d)
print("Best Arm:",Q.index(max(Q))+1)
