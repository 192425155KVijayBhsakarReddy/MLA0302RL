import pandas as pd
import random

data = pd.read_csv("P:\RL\student_marks.csv")

actions = ["Extra Study", "Revision", "Mock Test"]

q_online = [[0 for _ in range(3)] for _ in range(len(data))]
q_target = [[0 for _ in range(3)] for _ in range(len(data))]

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 100

# Training
for episode in range(episodes):

    for state in range(len(data) - 1):

        # Epsilon-Greedy Action Selection
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = q_online[state].index(max(q_online[state]))

        current_marks = data.loc[state, "Final_Marks"]
        next_marks = data.loc[state + 1, "Final_Marks"]

        # Reward
        if next_marks > current_marks:
            reward = 10
        elif next_marks == current_marks:
            reward = 2
        else:
            reward = -5

        next_state = state + 1

        # Double DQN Update
        best_action = q_online[next_state].index(max(q_online[next_state]))
        target = reward + gamma * q_target[next_state][best_action]

        q_online[state][action] += alpha * (target - q_online[state][action])

    # Update Target Network every 10 episodes
    if episode % 10 == 0:
        q_target = [row[:] for row in q_online]

# Display Results
print("Student Marks")
print(data[["Student_ID", "Final_Marks"]])

print("\nLearned Q-Table")
for i in range(len(data)):
    print("Student", data.loc[i, "Student_ID"], ":", [round(x, 2) for x in q_online[i]])

print("\nRecommended Action")
for i in range(len(data)):
    best = q_online[i].index(max(q_online[i]))
    print("Student", data.loc[i, "Student_ID"], "->", actions[best])
