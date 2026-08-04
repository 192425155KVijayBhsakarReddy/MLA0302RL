import pandas as pd
d=pd.read_csv("../Datasets/Q4.csv")
gamma=float(input("Gamma: "))
print("\nDelivery Points")
print(d)
order=d.sort_values("Type")
print("\nVisited Order")
print(order)
print("Gamma:",gamma)
