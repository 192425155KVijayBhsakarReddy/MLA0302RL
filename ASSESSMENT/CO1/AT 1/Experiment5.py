gamma=0.9
reward=[5,2,8]
value=[0,0,0]

for _ in range(10):
    new_value=[]
    for r,v in zip(reward,value):
        new_value.append(r+gamma*v)
    value=new_value

print("State Values:",value)
