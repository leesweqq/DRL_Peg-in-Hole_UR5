import os
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from rlenv import PegInHoleGymEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#train model 
def train(agent_name="ppo", total_timesteps=100_000, save_freq=10_000, save_path="./checkpoints/"):
    
    # Create a directory to store the log files if it doesn't exist
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the environment and wrap it with Monitor to log performance
    env = PegInHoleGymEnv()
    env = Monitor(env, log_dir)  # Monitor the environment and store logs in the specified directory

    # Choose the algorithm
    agent_name = agent_name.lower()
    if agent_name == "ppo":
        model_class = PPO
        policy = "MultiInputPolicy"
    elif agent_name == "sac":
        model_class = SAC
        policy = "MultiInputPolicy"
    elif agent_name == "a2c":
        model_class = A2C
        policy = "MultiInputPolicy"
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    # Create the model
    model = model_class(policy, env, verbose=1, device="cuda", tensorboard_log=log_dir)  

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=save_path,
                                             name_prefix=f"{agent_name}_model")

    # Train the model and save checkpoints
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save the final model after training
    model.save(os.path.join(save_path, f"{agent_name}_final_model"))

    return model, env

#test model 1000 times episode
def test_rl_model(agent_name):
    # Create the environment
    env = PegInHoleGymEnv()

    # Choose the algorithm
    if agent_name == "ppo":
        model_class = PPO
        policy = "MultiInputPolicy"
    elif agent_name == "sac":
        model_class = SAC
        policy = "MultiInputPolicy"
    elif agent_name == "a2c":
        model_class = A2C
        policy = "MultiInputPolicy"
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    # Load the trained model
    model = model_class(policy, env, verbose=1, device="cuda")
    model = model.load(f"./checkpoints/{agent_name}_model_160000_steps")

    success_count = 0
    failure_count = 0
    episode_count = 0
    max_episodes = 1000

    obs, info = env.reset()

    while episode_count < max_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            episode_count += 1

            if info.get("insertion_success", True):
                success_count += 1
                print(f"✅ Episode {episode_count}: Success")
            else:
                failure_count += 1
                print(f"❌ Episode {episode_count}: Failure")

            obs, info = env.reset() 

    success_rate = (success_count / max_episodes) * 100
    print(f"\n Test completed: {max_episodes} episodes in total")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Success Rate: {success_rate:.2f}%")




# Function to smooth the data using a moving average
def smooth(data, window_size=10):
    """Apply weighted moving average smoothing to the data"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


# Function to plot the training reward data
def plot_reward_data():
    # Set the directory for storing Monitor log files
    log_dir = "./logs"
    files = {
        "SAC": "monitor_sac.csv",
        "PPO": "monitor_ppo.csv",
        "A2C": "monitor_a2c.csv",
    }

    plt.figure(figsize=(8, 5))

    for label, file_name in files.items():
        monitor_file = os.path.join(log_dir, file_name)

        # Read the Monitor log data (skip the first row of comments)
        data = pd.read_csv(monitor_file, skiprows=1)

        # Calculate cumulative timestep
        data['timestep'] = np.cumsum(data['l'])
        data['timestep'] -= data['timestep'].iloc[0]

        # Apply smoothing to rewards
        smoothed_rewards = smooth(data['r'])
        smoothed_timesteps = data['timestep'][:len(smoothed_rewards)]

        # Filter to timesteps <= 160000
        mask = smoothed_timesteps <= 160000
        plt.plot(
            smoothed_timesteps[mask],
            smoothed_rewards[mask],
            label=label
        )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.legend(loc='lower right') 
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Choose agent_name = "sac" or "ppo" or "a2c"
    agent_name = "sac"

    # # Train the RL model
    # # train(agent_name=agent_name, total_timesteps=250000, save_freq=10000)

    # Test the trained RL model
    test_rl_model(agent_name)

    # # Plot the reward training graph
    # plot_reward_data()
