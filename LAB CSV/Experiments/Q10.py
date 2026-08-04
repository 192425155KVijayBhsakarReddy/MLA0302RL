import pandas as pd
d=pd.read_csv('../Datasets/Q10.csv')
lr=float(input('Learning Rate: '));ep=int(input('Episodes: '))
w=0
for _ in range(ep):
 for r in d['Return']: w+=lr*r
print(d);print('Policy Weight:',round(w,3));print('Average Return:',round(d['Return'].mean(),4))