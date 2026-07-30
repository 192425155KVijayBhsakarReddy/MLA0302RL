import random

epsilon=0.2
actions=["Left","Right"]

for i in range(20):
    if random.random()<epsilon:
        print("Explore:",random.choice(actions))
    else:
        print("Exploit: Right")
