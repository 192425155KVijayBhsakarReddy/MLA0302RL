import random

states=["A","B","C"]
rewards={"A":1,"B":2,"C":5}

current_state="A"

print("Markov Decision Process Simulation")

for i in range(10):
    print("\nCurrent State:",current_state)
    next_state=random.choice(states)
    print("Next State:",next_state)
    print("Reward:",rewards[next_state])
    current_state=next_state
