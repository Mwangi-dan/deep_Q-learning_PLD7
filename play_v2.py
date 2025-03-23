import gymnasium as gym
from stable_baselines3 import DQN
import ale_py

# Create environment with rendering enabled
env = gym.make("ALE/Galaxian-v5", render_mode="human")

# Load the trained model
try:
    model = DQN.load("models/model_dqns_mlp.zip")
except Exception as e:
    print(f"Error loading model: {e}\nMake sure you have run train.py first and the model file exists.")
    exit()

# Run episodes
for episode in range(10):
    obs, _ = env.reset()
    total_reward, steps, done = 0, 0, False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

env.close()
