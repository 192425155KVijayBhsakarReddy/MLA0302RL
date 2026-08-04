import pandas as pd
d=pd.read_csv('../Datasets/Q9.csv')
e=int(input('Episodes: '))
v=sum(d['Reward'].mean() for _ in range(e))/e
print(d);print('Estimated Value:',round(v,2))