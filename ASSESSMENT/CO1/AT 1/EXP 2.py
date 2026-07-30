import gymnasium as gym

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1")

# Reset the environment
state, info = env.reset()

print("Initial State:", state)

done = False
step = 1

while not done:

    # Select a random action
    action = env.action_space.sample()

    # Perform the action
    next_state, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", step)
    print("Current State:", state)
    print("Action:", action)
    print("Next State:", next_state)
    print("Reward:", reward)

    state = next_state
    step += 1

    done = terminated or truncated

print("\nEpisode Finished")

env.close()
