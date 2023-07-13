# 
# -*- coding: utf-8 -*-
import os
import gc
import sys
import json
import random

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
AGENTS_DIR = os.path.join(BASE_DIR,'agents')
sys.path.append(AGENTS_DIR)
from agents.PPO import PPO
from agents.PDPPO import PDPPO

from agents.PDPPOAgent import PDPPOAgent
from agents.PPOAgent import PPOAgent

import numpy as np
from envs import *
from agents import *
from agents import StochasticProgrammingAgent, AdpAgentHD3
from agents import StableBaselineAgent, MultiAgentRL, EnsembleAgent, PerfectInfoAgent,PSOagent,AdpAgentHD, PPOAgent
from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel


#'15items_5machines_i100','25items_10machines'

if __name__ == '__main__':
    experiments = ['15items_5machines_i100','20items_10machines','25items_10machines']
    for experiment_name in experiments:
        for i in range(0,5):
            # Setting the seeds
            np.random.seed(1)
            random.seed(10)
            # Environment setup load:
            # experiment_name = '15items_5machines_i100' # we set the experiment using the available files in cfg
            # experiment_name = '25items_10machines' # we set the experiment using the available files in cfg
            file_path = os.path.abspath(f"./cfg_env/setting_{experiment_name}.json")
            fp = open(file_path, 'r')
            settings = json.load(fp)
            fp.close()
            
            # Models setups:
            stoch_model = StochasticDemandModel(settings)
            settings['time_horizon'] = 100
            env = SimplePlant(settings, stoch_model)
            settings['dict_obs'] = False
            setting_sol_method = {
                'discount_rate': 0.99,
                'experiment_name': experiment_name,
                'parallelization': False,
                'model_name': 'A2C',
                'branching_factors': [4, 2, 2],
                'dict_obs': False # To be employed if dictionary observations are necessary
            }
            # Parameters for the ADPHS:
            setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
            setting_sol_method['discount_rate'] = 0.9
            agents = []
            # Parameters for the RL:
           
            training_epochs_RL = 30000
            training_epochs_multiagent = 2000
            
            
            setting_sol_method['parallelization'] = False
            env = SimplePlant(settings, stoch_model)
            
            # Number of test execution (number of complet environment iterations)
            nreps = 100
            
            ###########################################################################
            # #PPO
            ###########################################################################
            
            # base_model_name = 'PPO'
            # ppo_agent = PPOAgent(
            #     env,
            #     setting_sol_method
            # )
            # ppo_agent.learn(n_episodes=training_epochs_RL*settings['time_horizon'] ) # Each ep with 200 steps
            
            # #load best agent before appending in the test list
            # BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
            # ppo_agent.load_agent(BEST_MODEL_DIR) # For training purposes
            # agents.append(("PPO", ppo_agent))
            
            ###########################################################################
            # Post-decision PPO
            ###########################################################################
            
            base_model_name = 'PDPPO'
            pdppo_agent = PDPPOAgent(
                env,
                setting_sol_method
            )
            pdppo_agent.learn(n_episodes=training_epochs_RL*settings['time_horizon'] ) # Each ep with 200 steps
            
            #load best agent before appending in the test list
            BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
            pdppo_agent.load_agent(BEST_MODEL_DIR) # For training purposes
            agents.append(("PDPPO", pdppo_agent))
            
            
            ###########################################################################
            #TESTING
            settings['dict_obs'] = False
            setting_sol_method['multiagent'] = False
            setting_sol_method['dict_obs'] = False
            env = SimplePlant(settings, stoch_model)
            setting_sol_method['experiment_name'] = experiment_name
            dict_res = test_agents(
                env,
                agents=agents,
                n_reps=nreps,
                setting_sol_method = setting_sol_method,
                use_benchmark_PI=False
            )
            
            for key,_ in agents:
                cost = dict_res[key,'costs']
                print(f'\n Cost in {nreps} iterations for the model {key}: {cost}')
            try:
                cost = dict_res['PI','costs']
                print(f'\n Cost in {nreps} repetitions for the model PI: {cost}')
            except:
                pass
                    
            #del multiagent
            del env
            gc.collect()
