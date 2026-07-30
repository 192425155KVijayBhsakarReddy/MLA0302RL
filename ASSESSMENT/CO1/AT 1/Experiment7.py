import numpy as np
import random

arms=5
counts=np.zeros(arms)
values=np.zeros(arms)
epsilon=0.1

for _ in range(100):
    if random.random()<epsilon:
        arm=random.randint(0,arms-1)
    else:
        arm=np.argmax(values)

    reward=np.random.rand()
    counts[arm]+=1
    values[arm]+= (reward-values[arm])/counts[arm]

print("Estimated Values:",values)
