import gymnasium as gym
import numpy as np
import os
import time
from ale_py import ALEInterface
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper

os.environ["ALE_ROMS"] = os.path.abspath("./roms")
ALEInterface()


def make_atari_env(env_id, n_envs=1, seed=None, render_mode="human"):
    def make_env(rank):
        def _init():
            env = gym.make(env_id, render_mode=render_mode)

            env = AtariWrapper(env)

            if seed is not None:
                env.reset(seed=seed + rank)
            return env

        return _init

    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)

    return env


def play_and_evaluate(model_path, env_id, num_episodes=5):
    model = DQN.load(model_path, device="cpu")
    env = make_atari_env(env_id, render_mode="human")

    for episode in range(num_episodes):
        obs = env.reset()
        done = [False]
        total_reward = 0

        while not done[0]:
            action, = model.predict(obs, deterministic=True)

            obs, reward, done = env.step(action)
            total_reward += reward[0]

            env.render()
            time.sleep(0.02)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    MODEL_PATH = "dqn_model.zip"
    ENV_ID = "ALE/Galaxian-v5"
    play_and_evaluate(MODEL_PATH, ENV_ID)
