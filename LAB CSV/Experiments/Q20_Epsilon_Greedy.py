import pandas as pd
import random

def load_content_dataset():
    return pd.read_csv('../Datasets/Q20.csv')

def epsilon_greedy_recommendation(dataset, rounds, epsilon):
    selection_count = [0] * len(dataset)
    reward_sum = [0] * len(dataset)

    for current_round in range(rounds):
        if random.random() < epsilon:
            selected_index = random.randint(0, len(dataset) - 1)
        else:
            average_rewards = [
                reward_sum[i] / selection_count[i] if selection_count[i] > 0 else 0
                for i in range(len(dataset))
            ]
            selected_index = average_rewards.index(max(average_rewards))

        probability = dataset.loc[selected_index, 'ClickProbability']
        reward = 1 if random.random() < probability else 0

        selection_count[selected_index] += 1
        reward_sum[selected_index] += reward

    dataset['Selections'] = selection_count
    dataset['TotalReward'] = reward_sum
    dataset['AverageReward'] = [
        round(reward_sum[i] / selection_count[i], 2) if selection_count[i] > 0 else 0
        for i in range(len(dataset))
    ]
    return dataset

def display_results(dataset):
    print('\n========== EPSILON GREEDY RESULT ==========')
    print(dataset)
    best = dataset.loc[dataset['AverageReward'].idxmax()]
    print('\nBest Content ID :', best['ContentID'])
    print('Average Reward :', best['AverageReward'])
    print('Total Selections :', best['Selections'])

def main():
    print('=' * 55)
    print(' EPSILON GREEDY CONTENT RECOMMENDATION ')
    print('=' * 55)

    content_dataset = load_content_dataset()

    print('\nLoaded Dataset')
    print(content_dataset)

    total_rounds = int(input('\nEnter Number of Rounds: '))
    epsilon_value = float(input('Enter Epsilon Value: '))

    result_dataset = epsilon_greedy_recommendation(
        content_dataset,
        total_rounds,
        epsilon_value
    )

    display_results(result_dataset)

if __name__ == '__main__':
    main()
