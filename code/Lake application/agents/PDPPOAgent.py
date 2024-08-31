import os # Provides a way of interacting with the file system
import sys
import glob # Helps find all the pathnames matching a specified pattern according to the rules used by the Unix shell
import time # Provides various time-related functions
from datetime import datetime # Module that supplies classes for working with dates and times

import numpy as np # A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices
import gym # Provides a collection of test problems — environments — that you can use to work out your reinforcement learning algorithms
import torch # A machine learning framework that provides tensor computation (like NumPy) with strong acceleration on GPUs
import copy # Provides a module for shallow and deep copying operations
import matplotlib.pyplot as plt # A plotting library for the Python programming language and its numerical mathematics extension NumPy
import matplotlib.patches as mpatches # Provides a way of adding a colored patch to the plot, for example to create a legend
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
AGENTS_DIR = os.path.join(BASE_DIR,'agents')
sys.path.append(AGENTS_DIR)
from agents.PDPPO import PDPPO
from envs import *
import copy


class PDPPOAgent():
    def __init__(self, env, settings: dict):
        self.env = env
        self.model_name = settings['model_name']
        self.experiment_name = settings['experiment_name']
        self.parallelization = settings['parallelization']
        try:self.dict_obs = settings['dict_obs']
        except:self.dict_obs = False
        
        self.POSSIBLE_STATES = self.env.observation_space.n
        self.env.cost_to_reward = True
        self.epsilon = 0

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
        # Use the logs file in the root path of the main.
        self.LOG_DIR = os.path.join(BASE_DIR,'logs')
        
        print("============================================================================================")
    
        ####### initialize environment hyperparameters ######
   
        self.has_continuous_action_space = False  # continuous action space; else discrete
    
        self.max_ep_len = 100                  # max timesteps in one episode
        self.tau = 1
        self.tau_start = 1.0  # initial value of tau
        self.tau_end = 2.0  # final value of tau
        
        self.print_freq = self.max_ep_len * 4        # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 4           # log avg reward in the interval (in num timesteps)
        self.save_model_freq = int(4999)          # save model frequency (in num timesteps)
    
        self.action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.05        # linearly decay self.action_std (self.action_std = self.action_std - self.action_std_decay_rate)
        self.min_action_std = 0.1                # minimum self.action_std (stop decay after self.action_std <= min_self.action_std)
        self.action_std_decay_freq = int(2.5e5)  # self.action_std decay frequency (in num timesteps)
        #####################################################
    
        ## Note : print/log frequencies should be > than self.max_ep_len
    
        ################ PDPPO hyperparameters ################
        self.update_timestep = int(self.max_ep_len*5)      # update policy every n timesteps
        self.K_epochs = 50               # update policy for K epochs in one PDPPO update
        self.buffer_size_mul = 5        # buffer size multiplier

        self.eps_clip = 0.2          # clip parameter for PDPPO
        self.gamma = 0.90            # discount factor
    
        self.lr_actor = 0.00055       # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network
    
        self.random_seed = 0         # set random seed if required (0 = no random seed)
        #####################################################
        self.run_num_pretrained = 0      #### change this to prevent overwriting weights in same self.experiment_name folder
        
        print("training environment name : " + self.experiment_name + '_PDPPO')
    
        # state space dimension
        self.state_dim = self.env.observation_space.n
    
        # action space dimension
        if self.has_continuous_action_space:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.n

        self.pdppo_agent = PDPPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip, copy.copy(self.env), self.has_continuous_action_space,self.tau, self.action_std)

       
    ################################### Training ###################################
    def learn(self,n_episodes = 100000):

    
        ###################### logging ######################
        
        self.max_training_timesteps = n_episodes   # break training loop if timeteps > self.max_training_timesteps
        
        env = self.env
        
        #### log files for multiple runs are NOT overwritten
        log_dir = self.LOG_DIR
        if not os.path.exists(log_dir):
              os.makedirs(log_dir)
    
        log_dir = log_dir + '/' + self.experiment_name + '_PDPPO/'
        if not os.path.exists(log_dir):
              os.makedirs(log_dir)
    
        #### get number of log files in log directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
    
        #### create new log file for each run
        log_f_name = log_dir + '/PDPPO_' + self.experiment_name + "_log_" + str(run_num) + ".csv"
    
        print("current logging run number for " + self.experiment_name + " : ", run_num)
        print("logging at : " + log_f_name)
        #####################################################
    
        ################### checkpointing ###################
        
    
        directory = self.LOG_DIR
        if not os.path.exists(directory):
              os.makedirs(directory)
    
        directory = directory + '/' + self.experiment_name + '_PDPPO' + '/'
        if not os.path.exists(directory):
              os.makedirs(directory)
    
        
        checkpoint_path = directory + "PDPPO_{}_{}_{}.pth".format(self.experiment_name, self.random_seed, self.run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)
        #####################################################
    
    
        ############# print all hyperparameters #############
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", self.max_training_timesteps)
        print("max timesteps per episode : ", self.max_ep_len)
        print("model saving frequency : " + str(self.save_model_freq) + " timesteps")
        print("log frequency : " + str(self.log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(self.print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", self.state_dim)
        print("action space dimension : ", self.action_dim)
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", self.action_std)
            print("decay rate of std of action distribution : ", self.action_std_decay_rate)
            print("minimum std of action distribution : ", self.action_std)
            print("decay frequency of std of action distribution : " + str(self.action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PDPPO update frequency : " + str(self.update_timestep) + " timesteps")
        print("PDPPO K epochs : ", self.K_epochs)
        print("PDPPO epsilon clip : ", self.eps_clip)
        print("discount factor (self.gamma) : ", self.gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", self.lr_actor)
        print("optimizer learning rate critic : ", self.lr_critic)
        if self.random_seed:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", self.random_seed)

        #####################################################
    
        print("============================================================================================")
    
        ################# training procedure ################
    
        # initialize a PDPPO agent
        self.PDPPO_agent = PDPPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs, self.eps_clip, copy.copy(self.env), self.has_continuous_action_space, self.action_std)
    
        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
    
        print("============================================================================================")
    
        # logging file
        log_f = open(log_f_name,"w+")
        log_f.write('episode,timestep,reward\n')
    
        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0
    
        log_running_reward = 0
        log_running_episodes = 0
    
        time_step = 0
        i_episode = 0
        
        annealing_steps = self.max_training_timesteps  # total number of training steps
        
        # training loop
        while time_step <= self.max_training_timesteps:
            
            anneal_rate = (self.tau_end - self.tau_start) / annealing_steps  # rate of tau increase per step
            
            self.tau = max(self.tau_end, self.tau_start + anneal_rate * time_step)
            
            state = env.reset()
            current_ep_reward = 0
    
            binary_array = np.zeros(self.state_dim, dtype=int)
            binary_array[state] = 1
            state = binary_array

            for t in range(1, self.max_ep_len+1):
                # select action with policy
                action, post_reward = self.pdppo_agent.select_action(state,self.tau)
                state, reward, done, _ = env.step(action.item())

                binary_array = np.zeros(self.state_dim, dtype=int)
                binary_array[state] = 1
                state = binary_array

                # saving reward and is_terminals
                self.pdppo_agent.buffer.rewards.append(reward - post_reward.item()) # we save the reward without the post decision reward
                self.pdppo_agent.buffer.is_terminals.append(done)
    
                time_step +=1
                current_ep_reward += reward
    
                # update PDPPO agent
                if time_step % self.update_timestep == 0:
                    self.pdppo_agent.update()
                
                    if time_step > self.update_timestep*self.buffer_size_mul:
                        self.pdppo_agent.buffer.clear(self.update_timestep)
    
                # if continuous action space; then decay action std of ouput action distribution
                if self.has_continuous_action_space and time_step % self.action_std_decay_freq == 0:
                    self.pdppo_agent.decay_self.action_std(self.action_std_decay_rate, self.action_std)
    
                # log in logging file
                if time_step % self.log_freq == 0:
    
                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)
    
                    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    log_f.flush()
    
                    log_running_reward = 0
                    log_running_episodes = 0
    
                # printing average reward
                if time_step % self.print_freq == 0:
    
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
    
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
    
                    print_running_reward = 0
                    print_running_episodes = 0
    
                # save model weights
                if time_step % self.save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    #print("saving model at : " + checkpoint_path)
                    self.pdppo_agent.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")
    
                # break; if the episode is over
                if done:
                    break
    
            print_running_reward += current_ep_reward
            print_running_episodes += 1
    
            log_running_reward += current_ep_reward
            log_running_episodes += 1
    
            i_episode += 1
    
        log_f.close()
        #env.close()
    
        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

    def load_agent(self,path):
        #directory = "PDPPO_preTrained" + '/' + env_name + '/'
        directory = self.LOG_DIR
        directory = directory + '/' + self.experiment_name + '_PDPPO' + '/'
        checkpoint_path = directory + "PDPPO_{}_{}_{}.pth".format(self.experiment_name, self.random_seed, self.run_num_pretrained)
        print("loading network from : " + checkpoint_path)
        self.pdppo_agent.load(checkpoint_path)