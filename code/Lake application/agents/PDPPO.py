
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

#print("============================================================================================")

# set device to cpu or cuda

device = torch.device('cpu')

# if(torch.cuda.is_available()): 

#     device = torch.device('cuda:0') 

#     torch.cuda.empty_cache()

#     print("Device set to : " + str(torch.cuda.get_device_name(device)))

# else:

#     print("Device set to : cpu")

#print("============================================================================================")





class NegReLU(nn.Module):

    def forward(self, x):

        return -torch.relu(x)



################################## PDPPO Policy ##################################

class RolloutBuffer:

    def __init__(self):

        self.actions = []

        self.states = []

        self.post_states = []

        self.logprobs = []

        self.rewards = []

        self.post_rewards = []

        self.state_values = []

        self.state_values_post = []

        self.is_terminals = []

    

    def clear(self,lag):

        self.actions = self.actions[lag:]
        self.states = self.states[lag:]
        self.post_states = self.post_states[lag:]
        self.logprobs = self.logprobs[lag:]
        self.rewards = self.rewards[lag:]
        self.post_rewards = self.post_rewards[lag:]
        self.state_values = self.state_values[lag:]
        self.state_values_post = self.state_values_post[lag:]
        self.is_terminals = self.is_terminals[lag:]

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
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 128),
                            nn.Tanh(),
                            nn.Linear(128, action_dim),
                            nn.Tanh()
                        )

        else:

            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 128),
                            nn.Tanh(),
                            nn.Linear(128, action_dim)
                        )


        # critic

        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1),

                    )

        

        self.critic_post = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1),
                    )

    

    def _initialize_actor(self, m):
        if isinstance(m, nn.Linear):
            # Example: Kaiming initialization for actor layers
            init.kaiming_uniform_(m.weight, nonlinearity='tanh')
            if m.bias is not None:
                init.zeros_(m.bias)



    def _initialize_critic(self, m):
        if isinstance(m, nn.Linear):
            # Example: Xavier initialization for critic layers
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, state):

        raise NotImplementedError


    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            #x = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(state))))
            logits = self.actor(state)
            action_probs = nn.functional.softmax(logits, dim=-1)  
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, post_state, action):

        #x = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(state))))
        logits = self.actor(state)
        action_probs = nn.functional.softmax(logits, dim=-1)  
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action.T).T
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
        self.policy.actor.apply(self.policy._initialize_actor)
        self.policy.critic.apply(self.policy._initialize_critic)
        self.policy.critic_post.apply(self.policy._initialize_critic)

        self.optimizer_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        self.optimizer_critic_post = torch.optim.Adam(self.policy.critic_post.parameters(), lr=lr_critic)

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
    
    def select_action(self, state, tau):
        state_int = state.copy()

        with torch.no_grad():
            state = torch.tensor(state).to(device)
            state = state.float() 
            state = torch.unsqueeze(state, 1).T
            action, action_logprob = self.policy_old.act(state)
        
        post_state, post_reward = self.env.get_post_decision_state(np.argmax(state_int.copy()),action.clone().cpu().numpy())

        binary_array = np.zeros(state.shape[1], dtype=int)
        binary_array[post_state] = 1

        post_state = torch.tensor(binary_array).to(device)
        post_state = post_state.float() 
        post_state = torch.unsqueeze(post_state, 1).T
        

        self.buffer.states.append(state)
        self.buffer.post_states.append(post_state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.post_rewards.append(post_reward)
        
        with torch.no_grad():
            state_val = self.policy_old.critic(state)
            state_val_post = self.policy_old.critic(post_state)            
        
        self.buffer.state_values.append(state_val)
        self.buffer.state_values_post.append(state_val_post)

        return action.cpu().numpy(), post_reward
    
    def update(self):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        post_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.post_rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            post_rewards.insert(0, discounted_reward)
            

        post_rewards = torch.tensor(post_rewards, dtype=torch.float32).to(device)

        # Normalizing the rewards

        # post_rewards = (post_rewards - post_rewards.mean()) / (post_rewards.std() + 1e-7)

        # pre_rewards = (rewards - post_rewards)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_post_states = torch.squeeze(torch.stack(self.buffer.post_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_state_values_post = torch.squeeze(torch.stack(self.buffer.state_values_post, dim=0)).detach().to(device)

        # Calculate advantages for current and subsequent states
        advantages_current = rewards - old_state_values
        advantages_post = post_rewards - old_state_values_post

        advantages = torch.max(advantages_current, advantages_post)
        

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, post_state_values, dist_entropy,  = self.policy.evaluate(old_states, old_post_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages.unsqueeze(1)

            critic_loss = self.MseLoss(state_values.squeeze(), rewards)

            critic_loss_post = self.MseLoss(post_state_values.squeeze(), post_rewards)

            actor_loss = (-torch.min(surr1, surr2) - 0.001 * dist_entropy).mean() + 0.7*(critic_loss.detach() + critic_loss_post.detach())

            # Update the actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # Update the critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            # Update the critic_post
            self.optimizer_critic_post.zero_grad()
            critic_loss_post.backward()
            self.optimizer_critic_post.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))