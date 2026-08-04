import pandas as pd

def load_dataset():
    return pd.read_csv("../Datasets/Q11.csv")

def calculate_profit(data,episodes,learning_rate,discount_factor):
    total_estimated_profit=0.0
    for current_episode in range(episodes):
        for index in range(1,len(data)):
            previous_price=data.loc[index-1,"Close"]
            current_price=data.loc[index,"Close"]
            if current_price>previous_price:
                profit=current_price-previous_price
                total_estimated_profit+=profit*learning_rate*discount_factor
    return total_estimated_profit

stock_market_data=load_dataset()
print("\n===== STOCK DATA =====")
print(stock_market_data.head())

training_episode_count=int(input("Enter Episodes: "))
learning_rate=float(input("Enter Learning Rate: "))
discount_factor=float(input("Enter Gamma: "))

estimated_profit=calculate_profit(stock_market_data,training_episode_count,learning_rate,discount_factor)

print("\n===== RESULT =====")
print("Episodes :",training_episode_count)
print("Learning Rate :",learning_rate)
print("Discount Factor :",discount_factor)
print("Estimated Profit :",round(estimated_profit,2))
