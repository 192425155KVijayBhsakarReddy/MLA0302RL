import gymnasium as gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("===================================")
print("Reinforcement Learning Environment")
print("===================================")

print("TensorFlow Version :", tf.__version__)
print("NumPy Version      :", np.__version__)

env = gym.make("CartPole-v1")

print("\nGymnasium Environment Created Successfully!")

state, info = env.reset()

print("Initial State:")
print(state)

env.close()

print("\nEnvironment Verification Completed Successfully.")
