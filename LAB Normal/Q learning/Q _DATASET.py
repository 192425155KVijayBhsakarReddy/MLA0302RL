import pandas as pd
import random

data = pd.read_csv("stock_data.csv")
prices = data["Close"].tolist()

actions = ["Buy", "Sell", "Hold"]

q_online = [[0.0 for _ in range(3)] for _ in range(len(prices))]
q_target = [[0.0 for _ in range(3)] for _ in range(len(prices))]

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 100

total_profit = 0

for episode in range(episodes):

    holding = False
    buy_price = 0

    for state in range(len(prices) - 1):

        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            action = q_online[state].index(max(q_online[state]))

        reward = 0

        if action == 0:
            if not holding:
                holding = True
                buy_price = prices[state]

        elif action == 1:
            if holding:
                reward = prices[state] - buy_price
                total_profit += reward
                holding = False

        next_state = state + 1

        best_action = q_online[next_state].index(max(q_online[next_state]))

        target = reward + gamma * q_target[next_state][best_action]

        q_online[state][action] += alpha * (target - q_online[state][action])

    if episode % 10 == 0:
        q_target = [row[:] for row in q_online]

print("Closing Prices")
print(prices)

print("\nLearned Q Table")
for i in range(len(prices)):
    print("State", i, ":", [round(x, 2) for x in q_online[i]])

print("\nTotal Profit :", total_profit)

print("\nBest Action for Each State")
for i in range(len(prices)):
    best = q_online[i].index(max(q_online[i]))
    print("Day", i + 1, "Price =", prices[i], "->", actions[best])
