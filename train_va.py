import os
from ale_py import ALEInterface
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

# Hyperparameter sets to test
hp_sets = [
    {  # Default baseline
        "lr": 0.00025,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_frac": 0.1,
    },
    {  # Experiment 1 - Lower learning rate
        "lr": 0.0001,
        "gamma": 0.99,
        "batch_size": 64,
        "eps_start": 1.0,
        "eps_end": 0.1,
        "eps_decay_frac": 0.2,
    },
    {  # Experiment 2 - Higher gamma
        "lr": 0.0005,
        "gamma": 0.995,
        "batch_size": 128,
        "eps_start": 0.8,
        "eps_end": 0.05,
        "eps_decay_frac": 0.15,
    },
]

# Environment setup
os.environ["ALE_ROMS"] = "./roms"
ALEInterface()

# Run experiments
for i, hp in enumerate(hp_sets):
    print(f"\nRunning Experiment {i+1} with params: {hp}")

    # Create environment
    env = make_atari_env("ALE/Galaxian-v5", n_envs=1)
    env = VecFrameStack(env, n_stack=4)

    # Create model with current hyperparameters
    model = DQN(
        "CnnPolicy",
        env,
        verbose=2,
        learning_rate=hp["lr"],
        gamma=hp["gamma"],
        batch_size=hp["batch_size"],
        exploration_initial_eps=hp["eps_start"],
        exploration_final_eps=hp["eps_end"],
        exploration_fraction=hp["eps_decay_frac"],
        tensorboard_log=f"./logs/exp_{i+1}",
    )

    # Train and save
    model.learn(total_timesteps=100_000)
    model.save(f"galaxian_dqn_exp_{i+1}", exclude=["replay_buffer"])

    # Record observations here manually
    print(f"Experiment {i+1} completed!")
    print("Note down performance metrics from TensorBoard or training logs")
