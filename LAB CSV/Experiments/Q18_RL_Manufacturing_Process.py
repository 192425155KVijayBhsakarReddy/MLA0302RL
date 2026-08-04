import pandas as pd

def load_manufacturing_dataset():
    return pd.read_csv('../Datasets/Q18.csv')

def initialize_quality_score(dataset):
    dataset['EstimatedValue'] = dataset['Reward']
    return dataset

def optimize_machine_settings(dataset, episodes, learning_rate, discount_factor):
    total_reward = 0
    for current_episode in range(episodes):
        updated_values = []
        for _, machine_state in dataset.iterrows():
            current_reward = machine_state['Reward']
            current_value = machine_state['EstimatedValue']
            improved_value = current_reward + (learning_rate * discount_factor * current_value)
            updated_values.append(round(improved_value, 2))
            total_reward += current_reward
        dataset['EstimatedValue'] = updated_values
    return dataset, total_reward

def display_results(dataset, total_reward):
    print('\n========== MANUFACTURING RESULT ==========')
    print(dataset)
    best_setting = dataset.loc[dataset['EstimatedValue'].idxmax()]
    print('\nBest Machine Setting :', best_setting['MachineSetting'])
    print('Highest Estimated Value :', best_setting['EstimatedValue'])
    print('Overall Reward :', total_reward)

def main():
    print('=' * 50)
    print(' RL MANUFACTURING PROCESS OPTIMIZATION ')
    print('=' * 50)
    manufacturing_dataset = load_manufacturing_dataset()
    print('\nLoaded Dataset')
    print(manufacturing_dataset)
    episode_count = int(input('\nEnter Number of Episodes: '))
    learning_rate = float(input('Enter Learning Rate: '))
    discount_factor = float(input('Enter Discount Factor (Gamma): '))
    manufacturing_dataset = initialize_quality_score(manufacturing_dataset)
    final_dataset, accumulated_reward = optimize_machine_settings(
        manufacturing_dataset,
        episode_count,
        learning_rate,
        discount_factor
    )
    display_results(final_dataset, accumulated_reward)

if __name__ == '__main__':
    main()
