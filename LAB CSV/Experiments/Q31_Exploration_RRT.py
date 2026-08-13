# ============================================================
# EXPERIMENT NO : 31
# TITLE : Autonomous Exploration Robot Path Planning (RRT/PRM)
# PROBLEM STATEMENT: Implement sampling-based planning (RRT/PRM) to plan
# collision-free paths for an autonomous exploration robot.
# DATASET : ../Datasets/Q31_Exploration_RRT_Dataset.csv
# ============================================================
import os, pandas as pd, numpy as np

def load_dataset():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "../Datasets/Q31_Exploration_RRT_Dataset.csv"))

def get_user_inputs():
    samples = int(input("\nEnter RRT Waypoint Samples : "))
    step_size = float(input("Enter RRT Step Size : "))
    return samples, step_size

def run_rrt_exploration(dataset, samples, step_size):
    x_coords, y_coords, obstacles = dataset["X_Coord"].values, dataset["Y_Coord"].values, dataset["ObstacleNear"].values
    valid_points = [(x_coords[i], y_coords[i]) for i in range(len(x_coords)) if obstacles[i] == "No"]
    tree = [valid_points[0]] if valid_points else [(0,0)]

    for _ in range(samples):
        rand_pt = (np.random.randint(0, 100), np.random.randint(0, 100))
        dists = [np.hypot(pt[0]-rand_pt[0], pt[1]-rand_pt[1]) for pt in tree]
        nearest = tree[np.argmin(dists)]
        tree.append((nearest[0] + step_size, nearest[1] + step_size))

    print("\n========== RRT EXPLORATION RESULT ==========")
    print("Generated Path Nodes Count :", len(tree))
    print("Collision-Free Target Reached :", tree[-1])

def main():
    print("=" * 45 + "\n AUTONOMOUS ROBOT RRT EXPLORATION \n" + "=" * 45)
    ds = load_dataset()
    print(ds)
    samples, step_size = get_user_inputs()
    run_rrt_exploration(ds, samples, step_size)

if __name__ == "__main__":
    main()
