# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 00:43:49 2023

@author: leona
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PDPPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.post_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.state_values_post = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.post_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.state_values_post[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            
            self.action_dim = action_dim
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.actor = nn.Linear(128, self.action_dim.nvec.sum())
            
            
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1)
                    )
        
        self.critic_post = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1)
                    )
    
    def forward(self, state):
        raise NotImplementedError
    
    
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    
    def act(self, state,tau):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            #x = nn.functional.relu(self.fc(state))
            x = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(state))))
            logits = self.actor(x)
            # x[torch.isnan(x)] = 0
            action_probs = nn.functional.softmax(logits, dim=-1)
            #action_probs = torch.nan_to_num(action_probs, nan=1e-6)
            dist = Categorical(action_probs.view(len(self.action_dim.nvec),-1))
            # action_probs = self.actor(state)
            # dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state,post_state, action,tau):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            #x = nn.functional.relu(self.fc(state))
            x = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(state))))
            # x[torch.isnan(x)] = 0
            logits = self.actor(x)
            action_probs = nn.functional.softmax(logits, dim=-1)
            #action_probs = torch.nan_to_num(action_probs, nan=1e-6)
            # mask = torch.isnan(action_probs)
            # if torch.all(mask):
            #     logits = torch.abs(logits)
            #     action_probs = nn.functional.softmax(logits, dim=-1)
                
            dist = Categorical(action_probs.view(state.shape[0],len(self.action_dim.nvec),-1))
            # action_probs = self.actor(state)
            # dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        state_values_post = self.critic_post(post_state)
        
        return action_logprobs, state_values, state_values_post, dist_entropy


class PDPPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, env, has_continuous_action_space, tau, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init
        
        self.tau = tau
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                        {'params': self.policy.critic_post.parameters(), 'lr': lr_critic}
                    ], weight_decay=0.001)

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PDPPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PDPPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
    
    def get_post_state(self, action, machine_setup, inventory_level):
        setup_loss = np.zeros(self.env.n_machines, dtype=int)
        setup_costs = np.zeros(self.env.n_machines)
        # if we are just changing the setup, we use the setup cost matrix with the corresponding position given by the actual setup and the new setup
        for m in range(self.env.n_machines):   
            if action[m] != 0: # if the machine is not iddle
                # 1. IF NEEDED CHANGE SETUP
                if machine_setup[m] != action[m] and action[m] != 0:
                    setup_costs[m] = self.env.setup_costs[m][action[m] - 1] 
                    setup_loss[m] = self.env.setup_loss[m][action[m] - 1]
                machine_setup[m] = action[m]
                # 2. PRODUCTION
                production = self.env.machine_production_matrix[m][action[m] - 1] - setup_loss[m]
                inventory_level[action[m] - 1] += production
            else:
                machine_setup[m] = 0
        # return the new machine_setup_inventory_level and the setup_cost        
        return machine_setup, inventory_level, setup_costs 
    
    def select_action(self, state,tau):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state,tau)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state,tau)
            
            
            machine_setup, inventory_level, setup_cost = self.get_post_state(action, state[self.env.n_items:self.env.n_items+self.env.n_machines].clone(), state[0:self.env.n_items].clone())
            
            post_state = state.clone()
            post_state[self.env.n_items:self.env.n_items+self.env.n_machines] = machine_setup.clone()
            post_state[0:self.env.n_items] = inventory_level.clone()
            post_state = torch.FloatTensor(post_state).to(device)
            
            self.buffer.states.append(state)
            self.buffer.post_states.append(post_state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            
            with torch.no_grad():
                #post_state = torch.cat([post_state.clone(),state.clone()])
                state_val = self.policy_old.critic(state)
                state_val_post = self.policy_old.critic_post(post_state)            
            
            self.buffer.state_values.append(state_val)
            self.buffer.state_values_post.append(state_val_post)

            return action.numpy()
    
    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_post_states = torch.squeeze(torch.stack(self.buffer.post_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_state_values_post = torch.squeeze(torch.stack(self.buffer.state_values_post, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - torch.min(old_state_values.detach(), old_state_values_post.detach()).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, state_values_post, dist_entropy = self.policy.evaluate(old_states,old_post_states, old_actions,self.tau)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.unsqueeze(1)

            # final loss of clipped objective PDPPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(torch.min(state_values,state_values_post.squeeze()), rewards) - 0.012 * dist_entropy
            
            loss_numpy = loss.detach().numpy()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1)
            self.optimizer.step()
            
        # Copy new weights into old policy
        
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))