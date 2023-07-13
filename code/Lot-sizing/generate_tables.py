# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:30:32 2023

@author: leona
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




def get_max_rewards():
    experiment_names = ['15items_5machines_i100', '20items_10machines', '25items_10machines']
    methods = ['PDPPO', 'PPO', 'PDPPO1C']
    results = pd.DataFrame(columns=['Environment', 'Method', 'Max Reward', 'Standard Deviation'])

    for experiment_name in experiment_names:
        for method in methods:
            env_name = experiment_name
            max_rewards = []
            for run_num in range(1, 6):
                log_f_name = f'logs/5 runs results/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                max_reward = data['reward'].max()
                max_rewards.append(max_reward)
            mean_max_reward = np.mean(max_rewards)
            std_max_reward = np.std(max_rewards)
            results = results.append({'Environment': env_name, 'Method': method, 'Max Reward': mean_max_reward, 'Standard Deviation': std_max_reward}, ignore_index=True)

    return results
    
def get_first_rewards():
    experiment_names = ['15items_5machines_i100', '20items_10machines', '25items_10machines']
    methods = ['PDPPO', 'PPO', 'PDPPO1C']
    results = pd.DataFrame(columns=['Environment', 'Method', 'Max Reward', 'Standard Deviation'])

    for experiment_name in experiment_names:
        for method in methods:
            env_name = experiment_name
            max_rewards = []
            for run_num in range(1, 6):
                log_f_name = f'logs/5 runs results/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                reward_500000 = data[data['timestep'] == 500000]['reward'].values
                max_rewards.append(reward_500000)
            mean_max_reward = np.mean(max_rewards)
            std_max_reward = np.std(max_rewards)
            results = results.append({'Environment': env_name, 'Method': method, 'Max Reward': mean_max_reward, 'Standard Deviation': std_max_reward}, ignore_index=True)

    return results

def get_steps_reward_threshold():
    experiment_names = ['15items_5machines_i100', '20items_10machines', '25items_10machines']
    methods = ['PDPPO', 'PPO', 'PDPPO1C']
    reward_thresholds = [-1900, -5500, -3700]
    results = pd.DataFrame(columns=['Environment', 'Method', 'Steps', 'Standard Deviation'])

    for i, experiment_name in enumerate(experiment_names):
        for j, method in enumerate(methods):
            env_name = experiment_name
            reward_steps = []
            for run_num in range(1, 6):
                log_f_name = f'logs/5 runs results/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                reward_threshold_value = reward_thresholds[i]
                reward_steps.append(data[data['reward'] >= reward_threshold_value]['timestep'].iloc[0])
            mean_reward_steps = np.mean(reward_steps) if reward_steps else np.nan
            std_reward_steps = np.std(reward_steps) if reward_steps else np.nan
            results = results.append({'Environment': env_name, 'Method': method, 'Steps': mean_reward_steps, 'Standard Deviation': std_reward_steps}, ignore_index=True)

    return results
    

if __name__ == '__main__':

    print(get_max_rewards())
    print(get_first_rewards())
    print(get_steps_reward_threshold())
