# ============================================================
# EXPERIMENT NO : 33
# TITLE : Value-Equivalence Prediction Model for Portfolios
#
# PROBLEM STATEMENT:
# Implement a value-equivalence prediction model to estimate and compare the
# long-term performance of alternative investment portfolio allocations.
#
# DATASET :
# ../Datasets/Q33_Portfolio_Value_Equivalence_Dataset.csv
# ============================================================

import os
import pandas as pd
import numpy as np

def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "../Datasets/Q33_Portfolio_Value_Equivalence_Dataset.csv")
    return pd.read_csv(path)

def display_dataset(dataset):
    print("\n========== PORTFOLIO DATASET ==========")
    print(dataset)

def get_user_inputs():
    investment_years = int(input("\nEnter Investment Horizon Years : "))
    principal = float(input("Enter Initial Capital Amount ($) : "))
    return investment_years, principal

def predict_value_equivalence(dataset, investment_years, principal):
    names = dataset["PortfolioName"].tolist()
    returns = dataset["HistoricalReturn"].values
    vols = dataset["RiskVol"].values

    dataset["PredictedEquivalentValue"] = 0.0
    for i, name in enumerate(names):
        # Expected compounded value formula
        final_val = principal * ((1 + returns[i] - 0.5 * vols[i]**2) ** investment_years)
        dataset.loc[i, "PredictedEquivalentValue"] = round(final_val, 2)

    print("\n========== VALUE EQUIVALENCE RESULT ==========")
    print(dataset[["PortfolioName", "PredictedEquivalentValue"]])
    best = dataset.loc[dataset["PredictedEquivalentValue"].idxmax()]["PortfolioName"]
    print("\nOptimal Portfolio Strategy :", best)

def main():
    print("=" * 45)
    print(" PORTFOLIO VALUE-EQUIVALENCE MODEL ")
    print("=" * 45)
    dataset = load_dataset()
    display_dataset(dataset)
    investment_years, principal = get_user_inputs()
    predict_value_equivalence(dataset, investment_years, principal)

if __name__ == "__main__":
    main()
