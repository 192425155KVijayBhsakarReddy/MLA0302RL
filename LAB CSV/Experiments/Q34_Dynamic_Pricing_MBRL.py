# ============================================================
# EXPERIMENT NO : 34
# TITLE : Model-Based Dynamic Pricing Policy Optimization
#
# PROBLEM STATEMENT:
# Develop a predictive demand & price sensitivity model to dynamically adjust
# prices in response to market conditions using Model-Based RL.
#
# DATASET :
# ../Datasets/Q34_Dynamic_Pricing_MBRL_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q34_Dynamic_Pricing_MBRL_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== DYNAMIC PRICING DATASET ==========")
    print(dataset)

def get_user_inputs():
    market_factor = float(input("\nEnter Market Demand Factor (e.g. 1.2) : "))
    price_step = float(input("Enter Price Adjustment Step ($) : "))
    return market_factor, price_step

def optimize_dynamic_pricing(dataset, market_factor, price_step):
    tiers = dataset["Tier"].tolist()
    base_prices = dataset["BasePrice"].values
    elasticity = dataset["PriceElasticity"].values
    demands = dataset["MarketDemand"].values

    dataset["OptimizedPrice"] = 0.0
    dataset["ExpectedRevenue"] = 0.0

    for i in range(len(tiers)):
        p = base_prices[i] + price_step
        # Q(s,a) Model-Based Demand forecasting: Demand * (1 + Elasticity * %PriceChange)
        d_pred = demands[i] * market_factor * (1 + elasticity[i] * (price_step / base_prices[i]))
        rev = p * max(0, d_pred)
        dataset.loc[i, "OptimizedPrice"] = round(p, 2)
        dataset.loc[i, "ExpectedRevenue"] = round(rev, 2)

    print("\n========== DYNAMIC PRICING RESULT ==========")
    print(dataset[["Tier", "BasePrice", "OptimizedPrice", "ExpectedRevenue"]])

def main():
    print("=" * 45)
    print(" DYNAMIC PRICING MODEL-BASED RL ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    market_factor, price_step = get_user_inputs()
    optimize_dynamic_pricing(dataset, market_factor, price_step)

if __name__ == "__main__":
    main()
