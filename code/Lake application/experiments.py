# 
# -*- coding: utf-8 -*-
import os
import gc
import sys
import json
import random
import gym
import torch

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
AGENTS_DIR = os.path.join(BASE_DIR,'agents')
sys.path.append(AGENTS_DIR)

from agents.PDPPOAgent import PDPPOAgent
from agents.PPOAgent import PPOAgent

import numpy as np
from agents import *

#'15items_5machines_i100','25items_10machines'

if __name__ == '__main__':
    for i in range(27,32):
        # Setting the seeds
        np.random.seed(i)
        random.seed(i)
        torch.manual_seed(i)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(i)
            torch.cuda.manual_seed_all(i)  # if you are using multi-GPU.

        
        from gym.envs.toy_text.frozen_lake import generate_random_map

        # Models setups:
        env = gym.make('FrozenLake-v1', desc=generate_random_map(size=10), is_slippery=True)

        experiment_name = 'frozen_lake'

        setting_sol_method = {
            'discount_rate': 0.99,
            'experiment_name': experiment_name,
            'parallelization': False,
            'model_name': 'PPO',
            'branching_factors': [4, 2, 2],
            'dict_obs': False # To be employed if dictionary observations are necessary
        }
        # Parameters for the RL:

        setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
        setting_sol_method['discount_rate'] = 0.99
        setting_sol_method['run'] = i
        agents = []
    
        training_epochs_RL = 200000
        
        setting_sol_method['parallelization'] = False
        
        # Number of test execution (number of complet environment iterations)
        nreps = 100
        
        ###########################################################################
        # #PPO
        ###########################################################################
        
        base_model_name = 'PPO'
        ppo_agent = PPOAgent(
            env,
            setting_sol_method
        )
        ppo_agent.learn(n_episodes=training_epochs_RL) # Each ep with 200 steps
        
        #load best agent before appending in the test list
        BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
        ppo_agent.load_agent(BEST_MODEL_DIR) # For training purposes

        
        
        ###########################################################################
        # Post-decision PPO
        ###########################################################################
        
        base_model_name = 'PDPPO'
        pdppo_agent = PDPPOAgent(
            env,
            setting_sol_method
        )
        pdppo_agent.learn(n_episodes=training_epochs_RL) # Each ep with 200 steps
        
        #load best agent before appending in the test list
        BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
        pdppo_agent.load_agent(BEST_MODEL_DIR) # For training purposes
        
        
        ###########################################################################
        #TESTING
        # settings['dict_obs'] = False
        # setting_sol_method['multiagent'] = False
        # setting_sol_method['dict_obs'] = False
        # env = SimplePlant(settings, stoch_model)
        # setting_sol_method['experiment_name'] = experiment_name
        # dict_res = test_agents(
        #     env,
        #     agents=agents,
        #     n_reps=nreps,
        #     setting_sol_method = setting_sol_method,
        #     use_benchmark_PI=False
        # )
        
        # for key,_ in agents:
        #     cost = dict_res[key,'costs']
        #     print(f'\n Cost in {nreps} iterations for the model {key}: {cost}')
        # try:
        #     cost = dict_res['PI','costs']
        #     print(f'\n Cost in {nreps} repetitions for the model PI: {cost}')
        # except:
        #     pass
                
        #del multiagent
        del env
        gc.collect()
