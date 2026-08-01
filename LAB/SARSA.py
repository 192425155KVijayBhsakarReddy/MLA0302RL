import numpy as np
import random


class GridWorld:
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:
            x = max(x - 1, 0)
        elif action == 1:
            x = min(x + 1, self.height - 1)
        elif action == 2:
            y = max(y - 1, 0)
        elif action == 3:
            y = min(y + 1, self.width - 1)

        next_state = (x, y)

        if next_state in self.obstacles:
            reward = -10
            done = True
        elif next_state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done
def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, 4))

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            Q[state[0], state[1], action] += alpha * \
                (reward + gamma * Q[next_state[0], next_state[1],
                 next_action] - Q[state[0], state[1], action])

            state = next_state
            action = next_action

    return Q
def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        q_values = Q[state[0], state[1]]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
if __name__ == "__main__":

    width = 5
    height = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(2, 2), (3, 2)]
    env = GridWorld(width, height, start, goal, obstacles)

    episodes = 1000
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1

    Q = sarsa(env, episodes, alpha, gamma, epsilon)

    print("Learned Q-values:")
    print(Q)
