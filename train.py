import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import ale_py

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Create environments
env = DummyVecEnv([lambda: gym.make("ALE/Galaxian-v5")])
eval_env = DummyVecEnv([lambda: gym.make("ALE/Galaxian-v5")])

# Initialize the DQN agent
model = DQN("CnnPolicy", env, learning_rate=1e-3, buffer_size=10000, learning_starts=1000,
            batch_size=32, gamma=0.99, exploration_fraction=0.1, exploration_initial_eps=1.0,
            exploration_final_eps=0.05, train_freq=4, target_update_interval=1000, verbose=1,
            tensorboard_log="logs/")

# Callbacks for checkpointing and evaluation
callbacks = [
    CheckpointCallback(save_freq=10000, save_path="models/", name_prefix="dqn_galaxian"),
    EvalCallback(eval_env, best_model_save_path="models/best_model", log_path="logs/",
                 eval_freq=10000, deterministic=True, render=False)
]

# Train the model
model.learn(total_timesteps=50000, callback=callbacks, progress_bar=True)

# Save the final model
model.save("models/model_dqns_mlp.zip")
print("Training completed! Model saved as 'mlp_dqn.zip'")
