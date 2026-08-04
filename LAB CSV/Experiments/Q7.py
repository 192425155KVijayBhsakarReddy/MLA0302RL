import pandas as pd
d=pd.read_csv('../Datasets/Q7.csv')
g=float(input('Gamma: '))
d['Value']=d['Reward']
for _ in range(5): d['Value']=d['Reward']+g*d['Value']
print(d[['NodeID','Type','Value']]);print('Best:',d.loc[d['Value'].idxmax(),'NodeID'])