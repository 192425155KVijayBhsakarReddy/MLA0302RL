import pandas as pd
d=pd.read_csv('../Datasets/Q8.csv')
p=input('Policy(Safe/Fast): ')
print(d)
print(d[d['Signal']=='Green'] if p.lower()=='safe' else d.sort_values('Distance'))