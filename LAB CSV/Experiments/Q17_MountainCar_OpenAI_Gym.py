import gym
import numpy as np

def create_environment():
    return gym.make("MountainCar-v0")

def run_policy(environment, total_episodes, max_steps):
    episode_rewards = []

    for current_episode in range(total_episodes):
        observation, _ = environment.reset()
        total_reward = 0

        for step in range(max_steps):
            action = environment.action_space.sample()
            observation, reward, terminated, truncated, _ = environment.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {current_episode+1} Reward : {total_reward}")

    return episode_rewards

def display_summary(rewards):
    print("\n========== MOUNTAIN CAR RESULT ==========")
    print("Episodes Executed :", len(rewards))
    print("Best Reward       :", max(rewards))
    print("Average Reward    :", round(sum(rewards)/len(rewards),2))
    print("Worst Reward      :", min(rewards))

def main():
    print("="*50)
    print(" OPENAI GYM MOUNTAIN CAR SIMULATION ")
    print("="*50)

    total_episodes = int(input("Enter Number of Episodes: "))
    maximum_steps = int(input("Enter Maximum Steps per Episode: "))

    environment = create_environment()

    reward_history = run_policy(
        environment,
        total_episodes,
        maximum_steps
    )

    environment.close()
    display_summary(reward_history)

if __name__ == "__main__":
    main()
