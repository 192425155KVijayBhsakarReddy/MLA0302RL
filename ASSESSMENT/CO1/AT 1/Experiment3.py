import gymnasium as gym

env = gym.make("FrozenLake-v1")

state, info = env.reset()

total_reward = 5
step = 1

print("----- Reinforcement Learning Framework -----")

while True:
    print("\nStep:", step)
    print("Current State:", state)

    action = env.action_space.sample()
    print("Action Taken:", action)

    next_state, reward, terminated, truncated, info = env.step(action)

    print("Next State:", next_state)
    print("Reward:", reward)

    total_reward += reward
    state = next_state
    step += 1

    if terminated or truncated:
        break

print("\nEpisode Finished")
print("Total Reward:", total_reward)

env.close()
