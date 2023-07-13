import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

main_folder = 'logs/results_2'

def get_max_rewards():
    experiment_names = ['frozen_lake']
    methods = ['PDPPO', 'PPO']
    results = pd.DataFrame(columns=['Environment', 'Method', 'Max Reward', 'Max Reward Standard Deviation'])
    main_folder = 'logs/results_2'
    for experiment_name in experiment_names:
        for method in methods:
            env_name = experiment_name
            max_rewards = []
            for run_num in range(1, 6):
                log_f_name = f'{main_folder}/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                max_reward = data['reward'].max()
                max_rewards.append(max_reward)
            mean_max_reward = np.mean(max_rewards)
            std_max_reward = np.std(max_rewards)
            results = results.append({'Environment': env_name, 'Method': method, 'Max Reward': mean_max_reward, 'Max Reward Standard Deviation': std_max_reward}, ignore_index=True)

    return results
    
def get_first_rewards():
    experiment_names = ['frozen_lake']
    methods = ['PDPPO', 'PPO']
    results = pd.DataFrame(columns=['Environment', 'Method', 'First Reward', 'First Reward Standard Deviation'])

    for experiment_name in experiment_names:
        for method in methods:
            env_name = experiment_name
            first_rewards = []
            for run_num in range(1, 6):
                log_f_name = f'{main_folder}/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                reward_50000 = data[data['timestep'] == 50000]['reward'].values
                first_rewards.append(reward_50000)
            mean_first_reward = np.mean(first_rewards)
            std_first_reward = np.std(first_rewards)
            results = results.append({'Environment': env_name, 'Method': method, 'First Reward': mean_first_reward, 'First Reward Standard Deviation': std_first_reward}, ignore_index=True)

    return results

def get_steps_reward_threshold():
    experiment_names = ['frozen_lake']
    methods = ['PDPPO', 'PPO']
    reward_thresholds = [10]
    results = pd.DataFrame(columns=['Environment', 'Method', 'Steps', 'Steps Standard Deviation'])

    for i, experiment_name in enumerate(experiment_names):
        for j, method in enumerate(methods):
            env_name = experiment_name
            reward_steps = []
            for run_num in range(1, 6):
                log_f_name = f'logs/results_1/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                reward_threshold_value = reward_thresholds[i]
                reward_steps.append(data[data['reward'] >= reward_threshold_value]['timestep'].iloc[0])
            mean_reward_steps = np.mean(reward_steps) if reward_steps else np.nan
            std_reward_steps = np.std(reward_steps) if reward_steps else np.nan
            results = results.append({'Environment': env_name, 'Method': method, 'Steps': mean_reward_steps, 'Steps Standard Deviation': std_reward_steps}, ignore_index=True)

    return results
    

if __name__ == '__main__':
    max_rewards_df = get_max_rewards()
    first_rewards_df = get_first_rewards()
    steps_rewards_df = get_steps_reward_threshold()

    final_results = pd.merge(max_rewards_df, first_rewards_df, on=['Environment', 'Method'])
    final_results = pd.merge(final_results, steps_rewards_df, on=['Environment', 'Method'])
    print(final_results)
