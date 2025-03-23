import os
from ale_py import ALEInterface
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

os.environ["ALE_ROMS"] = os.path.abspath("./roms")
os.environ["ALE_ROMS"] = "./roms"
ALEInterface()

env = make_atari_env("ALE/Galaxian-v5", n_envs=1)
env = VecFrameStack(env, n_stack=4)

model = DQN(
    "CnnPolicy",
    env,
    verbose=2,
    learning_rate=0.00005,
    gamma=0.99,
    batch_size=32,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.1,
    tensorboard_log="./logs/",
)

model.learn(total_timesteps=2_000_000)
model.save("dql_model", exclude=["replay_buffer"])
