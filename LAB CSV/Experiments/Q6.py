import pandas as pd,random,math
df=pd.read_csv('../Datasets/Q6.csv')
alg=input('Algorithm(Epsilon/UCB/Thompson): ')
r=int(input('Rounds: '));e=float(input('Epsilon: '))
Q=[0.0]*len(df);N=[0]*len(df)
for t in range(r):
 a=random.randrange(len(df)) if alg.lower()!='ucb' else max(range(len(df)),key=lambda i:Q[i]+math.sqrt(2*math.log(t+2)/(N[i]+1)))
 if alg.lower()=='epsilon' and random.random()>=e:a=Q.index(max(Q))
 rw=1 if random.random()<df.loc[a,'CTR'] else 0;N[a]+=1;Q[a]+=(rw-Q[a])/N[a]
print(df);print('Best Ad:',df.loc[Q.index(max(Q)),'Advertisement'])