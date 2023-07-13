# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:30:32 2023

@author: leona
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def save_graph():
    print("============================================================================================")
    
    # experiment_name = '15items_5machines_i100'
    # experiment_name = '20items_10machines'
    # experiment_name = '25items_10machines'
    experiment_name = 'frozen_lake'
    env_name = experiment_name
    
    rolling_window = 10

    BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
    BASE_DIR = BASE_DIR+'\\Post-Doctorate\\Lot-sizing\\Lake application'  

    # make directory for saving figures
    figures_dir = BASE_DIR + "\\results"
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # make environment directory for saving figures
    figures_dir = figures_dir + '/' + env_name + '_PPO'+'/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    #fig_save_path = figures_dir + '/PPO_' + env_name + '_fig_' + str(fig_num) + '.png'

    # get number of log files in directory
     
    # Use the logs file in the root path of the main.
    LOG_DIR = os.path.join(BASE_DIR,'logs')
    
    log_dir = LOG_DIR + '/' + env_name + '_PPO' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)-1

    all_runs_ppo = []
    
    
    ########################################################################################
    for run_num in range(num_runs):
        run_num = run_num + 1
        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)

        print("data shape : ", data.shape)

        all_runs_ppo.append(data)
        print("--------------------------------------------------------------------------------------------")

    # average all runs
    df_concat = pd.concat(all_runs_ppo)
    
    
    #Apply rolling mean to reward values
    df_concat['reward_mean'] = df_concat['reward'].rolling(window=rolling_window, win_type='triang', min_periods=1).mean()
    
    # Drop NaN values from beginning of rolling mean
    df_concat = df_concat.dropna().reset_index(drop=True)
    
    # Calculate mean and standard deviation of reward values
    reward_mean = df_concat.groupby('timestep')['reward_mean'].mean().iloc[rolling_window:]
    reward_std = df_concat.groupby('timestep')['reward_mean'].std().iloc[rolling_window:]
    
    # Set up plot using seaborn
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.set_style("whitegrid")
    # Plot mean reward with shaded confidence interval
    sns.lineplot(x=reward_mean.index, y=reward_mean, ax=ax,label='PPO')
    ax.fill_between(reward_mean.index, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    # keep only reward_smooth in the legend and rename it


    ########################################################################################
    
    
    log_dir = LOG_DIR + '/' + env_name + '_PDPPO' + '/'

    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)-1

    all_runs = []
    
    for run_num in range(num_runs):
        run_num = run_num + 1
        log_f_name = log_dir + 'PDPPO_' + env_name + "_log_" + str(run_num) + ".csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)

        print("data shape : ", data.shape)

        all_runs.append(data)
        print("--------------------------------------------------------------------------------------------")

    # average all runs
    df_concat = pd.concat(all_runs)
    
    #Apply rolling mean to reward values
    df_concat['reward_mean'] = df_concat['reward'].rolling(window=rolling_window, win_type='triang', min_periods=1).mean()
    
    # Drop NaN values from beginning of rolling mean
    df_concat = df_concat.dropna().reset_index(drop=True)
    
    # Calculate mean and standard deviation of reward values
    reward_mean = df_concat.groupby('timestep')['reward_mean'].mean().iloc[rolling_window:]
    reward_std = df_concat.groupby('timestep')['reward_mean'].std().iloc[rolling_window:]
    
    # Plot mean reward with shaded confidence interval
    sns.lineplot(x=reward_mean.index, y=reward_mean, ax=ax,label='PDPPO')
    ax.fill_between(reward_mean.index, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
    #ax.set(xlabel='Timestep', ylabel='Mean Reward', title='Average Reward with Confidence Interval')
    ax.legend()
    ########################################################################################
    
    # ax.set_yticks(np.arange(0, 1800, 200))
    # ax.set_xticks(np.arange(0, int(4e6), int(5e5)))

    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)

    fig = plt.gcf()

    fig.set_size_inches(15, 4)

    print("============================================================================================")
    fig.savefig(os.path.join(figures_dir, f'{experiment_name}.pdf'), dpi=300, bbox_inches='tight')
    print("figure saved at : ", figures_dir)
    print("============================================================================================")
    

if __name__ == '__main__':

    save_graph()
    
