import pandas as pd

def load_customer_dataset():
    return pd.read_csv('../Datasets/Q19.csv')

def monte_carlo_policy_evaluation(dataset, episode_count, discount_factor):
    estimated_returns=[]
    total_reward=0

    for current_episode in range(episode_count):
        episode_reward=0
        for _, customer in dataset.iterrows():
            reward=customer['Reward']
            episode_reward+=reward
        discounted_reward=episode_reward*discount_factor
        estimated_returns.append(round(discounted_reward,2))
        total_reward+=discounted_reward
        print(f'Episode {current_episode+1} Estimated Return : {round(discounted_reward,2)}')

    average_return=round(total_reward/episode_count,2)
    return estimated_returns,average_return

def display_summary(dataset,returns,average_return):
    print('\n========== CUSTOMER DATA ==========')
    print(dataset)
    print('\n========== MONTE CARLO RESULT ==========')
    print('Episodes Completed :',len(returns))
    print('Average Return     :',average_return)
    best_customer=dataset.loc[dataset['Reward'].idxmax()]
    print('Best Customer      :',best_customer['CustomerID'])
    print('Highest Reward     :',best_customer['Reward'])

def main():
    print('='*55)
    print(' MONTE CARLO POLICY EVALUATION - CUSTOMER CHURN ')
    print('='*55)

    customer_dataset=load_customer_dataset()

    episode_count=int(input('Enter Number of Episodes: '))
    discount_factor=float(input('Enter Discount Factor (Gamma): '))

    returns,average_return=monte_carlo_policy_evaluation(
        customer_dataset,
        episode_count,
        discount_factor
    )

    display_summary(customer_dataset,returns,average_return)

if __name__=='__main__':
    main()
