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
from scipy.stats import ttest_ind


def get_max_rewards(experiment_names, sample_size, methods):
    results = pd.DataFrame(columns=['Environment', 'Method', 'Max Reward', 'Standard Deviation'])

    for experiment_name in experiment_names:
        for method in methods:
            env_name = experiment_name
            max_rewards = []
            for run_num in range(1, sample_size+1):
                log_f_name = f'logs/{experiment_name}_{method}/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                max_reward = data['reward'].max()
                max_rewards.append(max_reward)
            mean_max_reward = np.mean(max_rewards)
            std_max_reward = np.std(max_rewards)
            results = results.append({'Environment': env_name, 'Method': method, 'Max Reward': mean_max_reward, 'Standard Deviation': std_max_reward}, ignore_index=True)

    return results
    
def get_first_rewards(experiment_names, sample_size, methods):
    results = pd.DataFrame(columns=['Environment', 'Method', 'Max Reward', 'Standard Deviation'])

    for experiment_name in experiment_names:
        for method in methods:
            env_name = experiment_name
            max_rewards = []
            for run_num in range(1, sample_size+1):
                log_f_name = f'logs/{experiment_name}_{method}/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                reward_500000 = data[data['timestep'] == 500000]['reward'].values
                max_rewards.append(reward_500000)
            mean_max_reward = np.mean(max_rewards)
            std_max_reward = np.std(max_rewards)
            results = results.append({'Environment': env_name, 'Method': method, 'First Reward': mean_max_reward, 'Standard Deviation': std_max_reward}, ignore_index=True)

    return results

def get_learning_metrics(experiment_names, sample_size=10, methods = ['PDPPO', 'PPO']):
    results = pd.DataFrame(columns=['Environment', 'Method', 'Max Reward', 'Standard Deviation MR', 'Cummulative Reward', 'Standard Deviation CR', 'Time to Threshold'])

    for experiment_name in experiment_names:
        for method in methods:
            env_name = experiment_name
            rewards_over_time = []
            max_rewards = []
            time_to_threshold = float('inf')
            threshold = 200  # Define your threshold based on your domain knowledge

            for run_num in range(1, sample_size + 1):
                log_f_name = f'logs/{experiment_name}_{method}/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                if not data.empty:
                    cum_rewards = data['reward'].cumsum()  # AUC calculation
                    if 'timestep' in data.columns:
                        first_above_threshold = data[data['reward'] >= threshold]['timestep'].min()
                        if pd.notna(first_above_threshold):
                            time_to_threshold = min(time_to_threshold, first_above_threshold)
                    max_rewards.append(data['reward'].max())
                    rewards_over_time.append(cum_rewards.iloc[-1])  # using the last value as AUC

            mean_max_reward = np.mean(max_rewards)
            std_max_reward = np.std(max_rewards)
            mean_auc = np.mean(rewards_over_time)
            std_auc = np.std(rewards_over_time)
            results = results.append({
                'Environment': env_name,
                'Method': method,
                'Max Reward': mean_max_reward,
                'Standard Deviation MR': std_max_reward,
                'Cummulative Reward': mean_auc,
                'Standard Deviation CR': std_auc,
                'Time to Threshold': time_to_threshold if time_to_threshold != float('inf') else None,
            }, ignore_index=True)

    return results

def get_steps_reward_threshold(experiment_names, sample_size, methods):
    reward_thresholds = [-1900, -5500, -3700]
    results = pd.DataFrame(columns=['Environment', 'Method', 'Steps', 'Standard Deviation'])

    for i, experiment_name in enumerate(experiment_names):
        for j, method in enumerate(methods):
            env_name = experiment_name
            reward_steps = []
            for run_num in range(1, sample_size+1):
                log_f_name = f'logs/{experiment_name}_{method}/{method}_{env_name}_log_{run_num}.csv'
                data = pd.read_csv(log_f_name)
                reward_threshold_value = reward_thresholds[i]
                reward_steps.append(data[data['reward'] >= reward_threshold_value]['timestep'].iloc[0])
            mean_reward_steps = np.mean(reward_steps) if reward_steps else np.nan
            std_reward_steps = np.std(reward_steps) if reward_steps else np.nan
            results = results.append({'Environment': env_name, 'Method': method, 'Steps': mean_reward_steps, 'Standard Deviation': std_reward_steps}, ignore_index=True)

    return results

def perform_t_tests(experiment_names, sample_size=10, metric='max', methods = ['PPO', 'PDPPO']):
    results = []

    for experiment_name in experiment_names:
        data_collection = {method: [] for method in methods}

        for method in methods:
            for run_num in range(1, sample_size + 1):
                env_name = experiment_name
                log_f_name = f'logs/{experiment_name}_{method}/{method}_{env_name}_log_{run_num}.csv'
                try:
                    data = pd.read_csv(log_f_name)
                    if 'reward' in data.columns:
                        if metric == 'max':
                            reward_value = data['reward'].max()  # Focus on max reward
                        elif metric == 'cumulative':
                            reward_value = data['reward'].sum()  # Focus on cumulative reward
                        else:
                            raise ValueError("Invalid metric type specified. Use 'max' or 'cumulative'.")
                        data_collection[method].append(reward_value)
                except FileNotFoundError:
                    print(f"File not found: {log_f_name}")
                except pd.errors.EmptyDataError:
                    print(f"No data in file: {log_f_name}")

        # Perform t-tests between each pair of methods
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1 = methods[i]
                method2 = methods[j]
                rewards1 = data_collection[method1]
                rewards2 = data_collection[method2]
                if rewards1 and rewards2:  # Ensure there is data to compare
                    t_stat, p_value = ttest_ind(rewards1, rewards2, equal_var=False)  # Using Welch's t-test for unequal variances
                    results.append((experiment_name, method1, method2, t_stat, p_value))
    
    # Print results
    for environment, method1, method2, t_stat, p_value in results:
        print(f"Environment: {environment}, Comparison: {method1} vs {method2}")
        print(f"  T-Statistic: {t_stat:.3f}, P-Value: {p_value:.3f}\n")


if __name__ == '__main__':
    #experiment_names = ['15items_5machines_t100_i100' , '20items_10machines_t100_i100' , '25items_10machines_t100_i100'] 
    experiment_names = ['20items_10machines_t100_i100' , '25items_10machines_t100_i100', '25items_15machines_t100_i100'] 
    #experiment_names = ['15items_5machines_t100_i100'] 
    methods = ['PDPPO', 'PPO', 'PDPPO_one_critic']
    iterations = 20
    # print(get_max_rewards(experiment_names, sample_size = iterations, methods=methods))
    # print(get_first_rewards(experiment_names, sample_size = iterations, methods=methods))
    print('===========Learning Metrics============')
    print(get_learning_metrics(experiment_names, iterations, methods=methods))
    methods = ['PDPPO', 'PPO']
    print('===========Maximums============')
    perform_t_tests(experiment_names, sample_size=iterations, metric='max', methods=methods)  # For maximum reward
    print('===========Cummulatives============')
    perform_t_tests(experiment_names, sample_size=iterations, metric='cumulative', methods=methods)  # For cumulative rewards
    #print(get_steps_reward_threshold())
