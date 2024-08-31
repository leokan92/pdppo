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

from agents.PDPPOAgent_one_critic import PDPPOAgent_one_critic
from agents.PPOAgent_two_critics import PPOAgent_two_critics
from agents.PDPPOAgent import PDPPOAgent
from agents.PPOAgent import PPOAgent
from agents.stableBaselineAgents import StableBaselineAgent


from test_functions import test_agents
import numpy as np
import torch
from envs import SimplePlant
from scenarioManager.stochasticDemandModel import StochasticDemandModel


#'15items_5machines_i100','25items_10machines'

if __name__ == '__main__':
    # experiments = ['15items_5machines_t100_i100']
    # experiments = ['20items_10machines_t100_i100']
    # experiments = ['25items_10machines_t100_i100']
    experiments = ['25items_15machines_t100_i100']
    #experiments = ['15items_5machines_t100_i100', '20items_10machines_t100_i100']
    for experiment_name in experiments:
        for i in range(20,21):
            # Setting the seeds
            np.random.seed(i)
            random.seed(i)
            torch.manual_seed(i)

            # If you are using CUDA (PyTorch with GPU support)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(i)
                torch.cuda.manual_seed_all(i)  # if you are using multi-GPU.

            # Environment setup load:
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
                'model_name': 'PPO',
                'branching_factors': [4, 2, 2],
                'dict_obs': False # To be employed if dictionary observations are necessary
            }
            # Parameters for the ADPHS:
            setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
            setting_sol_method['discount_rate'] = 0.99
            setting_sol_method['run'] = i
            agents = []
            # Parameters for the RL:
           
            training_epochs_RL = 10000 # 30000            

            env = SimplePlant(settings, stoch_model)
            
            # Number of test execution (number of complet environment iterations)
            nreps = 100
            
            ###########################################################################
            # Post-decision PPO - Dual critic
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
            # PPO
            ###########################################################################
            
            base_model_name = 'PPO'
            ppo_agent = PPOAgent(
                env,
                setting_sol_method
            )
            ppo_agent.learn(n_episodes=training_epochs_RL*settings['time_horizon'] ) # Each ep with 200 steps
            
            #load best agent before appending in the test list
            BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
            ppo_agent.load_agent(BEST_MODEL_DIR) # For training purposes
            agents.append(("PPO", ppo_agent))

            ###########################################################################
            # Post-decision PPO - Single Critic
            ###########################################################################
            
            base_model_name = 'PDPPO_one_critic'
            pdppo_agent_one_critic = PDPPOAgent_one_critic(
                env,
                setting_sol_method
            )
            pdppo_agent_one_critic.learn(n_episodes=training_epochs_RL*settings['time_horizon'] ) # Each ep with 200 steps
            
            #load best agent before appending in the test list
            BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
            pdppo_agent_one_critic.load_agent(BEST_MODEL_DIR) # For training purposes
            agents.append(("PDPPO_one_critic", pdppo_agent_one_critic))

            ###########################################################################
            # PPO - two critic
            ###########################################################################
            
            # base_model_name = 'PPO_two_critics'
            # ppo_agent_two_critics = PPOAgent_two_critics(
            #     env,
            #     setting_sol_method
            # )
            # ppo_agent_two_critics.learn(n_episodes=training_epochs_RL*settings['time_horizon'] ) # Each ep with 200 steps
            
            # #load best agent before appending in the test list
            # BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
            # ppo_agent_two_critics.load_agent(BEST_MODEL_DIR) # For training purposes
            # agents.append(("PPO_two_critics", ppo_agent_two_critics))

            ###########################################################################
            # RL A2C
            ###########################################################################

            # setting_sol_method['multiagent'] = False
            # setting_sol_method['parallelization'] = False

            # base_model_name = 'A2C'
            # setting_sol_method['parallelization'] = False
            # env = SimplePlant(settings, stoch_model)
            # setting_sol_method['model_name'] = base_model_name
            # rl_agent = StableBaselineAgent(
            #     env,
            #     setting_sol_method
            # )
            
            # rl_agent.learn(epochs=training_epochs_RL) # Each ep with 200 steps
            
            # #load best agent before appending in the test list
            # BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
            # rl_agent.load_agent(BEST_MODEL_DIR)
            # agents.append(("A2C", rl_agent))
                    
            
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
