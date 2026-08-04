import pandas as pd

grid_world_data=pd.read_csv("../Datasets/Q14.csv",header=None)

discount_factor=float(input("Enter Gamma: "))
iteration_count=int(input("Enter Iterations: "))

updated_value_table=grid_world_data.copy()

for iteration in range(iteration_count):
    updated_value_table=updated_value_table*discount_factor

print("\n===== GRID WORLD =====")

print(grid_world_data)

print("\n===== VALUE TABLE =====")

print(updated_value_table)

print("Completed",iteration_count,"iterations")
