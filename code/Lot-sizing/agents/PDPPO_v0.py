import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from envs import *
import gym



class SimplePlantSB(SimplePlant):
    def __init__(self, settings, stoch_model):
        super().__init__(settings, stoch_model)
        try:self.dict_obs = settings['dict_obs']
        except:self.dict_obs = False
        self.last_inventory = copy.copy(self.inventory_level)
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_items+1] * self.n_machines
        )
        
        if self.dict_obs:
            self.observation_space = gym.spaces.Dict({
                'inventory_level': gym.spaces.Box(low = np.zeros(self.n_items),high = np.ones(self.n_items)*(settings['max_inventory_level'][0]+1)*self.n_items),
                'machine_setup': gym.spaces.MultiDiscrete([self.n_items+1] * self.n_machines)
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=np.zeros(self.n_items+self.n_machines),# high for the inventory level
                high=np.concatenate(
                    [
                        np.array(self.max_inventory_level),
                        np.ones(self.n_machines) * (self.n_items+1), #high for the machine setups 
                    ]),
                dtype=np.int32
            )

    def step(self, action):
        """
        Step method: Execute one time step within the environment

        Parameters
        ----------
        action : action given by the agent

        Returns
        -------
        obs : Observation of the state give the method _next_observation
        reward : Cost given by the _reward method
        done : returns True or False given by the _done method
        dict : possible information for control to environment monitoring

        """
        self.last_inventory = copy.copy(self.inventory_level)
            
        self.total_cost = self._take_action(action, self.machine_setup, self.inventory_level, self.demand)
        
        # self.total_cost['setup_costs'] = 0
        # self.total_cost['holding_costs'] = 0
        
        reward = -sum([ele for key, ele in self.total_cost.items()])
        #reward = -self.total_cost['lost_sales']
        
        #reward = np.abs(action)
        
        self.current_step += 1
        done = self.current_step == self.T
        obs = self._next_observation()

        return obs, reward, done, self.total_cost

    def _next_observation(self):
        """
        Returns the next demand
        """
        obs = SimplePlant._next_observation(self)
        #obs['last_inventory_level'] = copy.copy(self.last_inventory)
        if isinstance(obs, dict):    
            if not self.dict_obs:
                obs = np.concatenate(
                    (
                        obs['inventory_level'], # n_items size
                        obs['machine_setup'], # n_machine size
                        #obs['last_inventory_level']# n_items size
                    )
                )
        else:
            if self.dict_obs:
                raise('Change dict_obst to False')
        return obs

# Define the policy network
class Policy(nn.Module):
    def __init__(self, input_size, output_shape):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc_list = nn.ModuleList([nn.Linear(128, output_shape[0]) for list(output_shape)[1] in range(0,output_shape[1])])
    
    def forward(self, x):
        x = F.relu(self.fc1(x)).requires_grad_()
        outputs = [F.softmax(fc(x), dim=1)for fc in self.fc_list]
        return outputs

# Define the value network for deterministic components
class Value(nn.Module):
    def __init__(self,input_size,output_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x)).requires_grad_()
        x = self.fc2(x)
        return x

# Define the value network for stochastic components
class ValueStochastic(nn.Module):
    def __init__(self,input_size,output_size):
        super(ValueStochastic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x)).requires_grad_()
        x = F.softmax(self.fc2(x), dim=1)
        return x

# Define the PPO agent
class PDPPO:
    def __init__(self, env: SimplePlant, settings: dict):
        
        self.env = SimplePlantSB(env.settings, env.stoch_model)
        self.last_inventory = env.inventory_level
        self.experiment_name = settings['experiment_name']
        try:self.dict_obs = settings['dict_obs']
        except:self.dict_obs = False
        
        self.POSSIBLE_STATES = self.env.n_items + 1
        self.env.cost_to_reward = True
        self.epsilon = 0

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
        # Use the logs file in the root path of the main.
        self.LOG_DIR = os.path.join(BASE_DIR,'logs')
        
        
        if self.dict_obs == False:
            input_size = self.env.observation_space.shape[0]
        output_size_policy = (self.env.n_items+1, self.env.action_space.shape[0]) # we add 1 for the idle state
        output_size_value = self.env.action_space.shape[0]
        self.policy = Policy(input_size,output_size_policy)
        self.value = Value(input_size,output_size_value)
        self.value_post = ValueStochastic(input_size,output_size_value)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=1e-3)
        self.optimizer_value_post = optim.Adam(self.value_post.parameters(), lr=1e-3)
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95

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

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        probs_concat = torch.stack(probs, dim=1)
        m = Categorical(probs_concat)
        action = m.sample()
        value = self.value(state)
        machine_setup, inventory_level, setup_cost = self.get_post_state(action.numpy()[0], state[0][self.env.n_items:self.env.n_items+self.env.n_machines].numpy(), state[0][0:self.env.n_items].numpy())
        value_post = self.value_post(state)
        
        return action, m.log_prob(action), probs_concat, value, value_post
    
    
    def update(self, rewards, rewards_pre_state, rewards_post_state, states, post_states, actions, probs, next_states):
        # Update deterministic value function
        for epoch in range(10):
            for i in range(len(actions)):
                state = torch.from_numpy(states[i]).float().unsqueeze(0)
                value = self.value(state)
                next_state = torch.from_numpy(next_states[i]).float().unsqueeze(0)
                next_value = self.value(next_state)
                target = rewards_pre_state[i] + self.gamma * next_value
                advantage = target - value
                loss = advantage.pow(2).mean()
                self.optimizer_value.zero_grad()
                loss.backward()
                self.optimizer_value.step()

        # Update stochastic value function
        for epoch in range(10):
            for i in range(len(actions)):
                state = torch.from_numpy(states[i]).float().unsqueeze(0)
                value = self.value_post(state)
                post_state = torch.from_numpy(post_states[i]).float().unsqueeze(0)
                value_post = self.value_post(post_state)
                target = rewards_post_state[i] + self.gamma * value_post
                advantage = target - value
                loss = advantage.pow(2).mean()
                self.optimizer_value_post.zero_grad()
                loss.backward()
                self.optimizer_value_post.step()

        # Update policy network
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.cat(actions).unsqueeze(1)
        old_probs = torch.cat(probs)
        old_probs = torch.gather(old_probs.clone(),2, actions)

        policy_epochs = 10
        for epoch in range(policy_epochs):
            probs = self.policy(states)
            probs = torch.stack(probs, dim=1).clone()
            m = Categorical(probs)
            action = m.sample()
            probs = torch.gather(probs, 2, actions)
            kl_div = (old_probs * (torch.log(old_probs) - torch.log(probs))).sum()

            for state,post_state, action, old_prob, prob, next_state, reward_pre_state, reward_post_state in zip(states,post_states, actions, old_probs, probs, next_states,rewards_pre_state,rewards_post_state):
                state = state.unsqueeze(0)
                next_state = torch.from_numpy(next_state).unsqueeze(0).float()
                post_state = torch.from_numpy(post_state).unsqueeze(0).float()
                action = action.unsqueeze(0)
                old_prob = old_prob.unsqueeze(0)
                prob = prob.unsqueeze(0)
                value = self.value(state)
                value_post = self.value_post(post_state)
                advantage = reward_pre_state + self.gamma * self.value(next_state) - self.value(state)
                advantage_post = reward_post_state + self.gamma * self.value_post(post_state) - self.value_post(state)
                
                ratio = (prob / old_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                policy_loss = -torch.min(surr1, surr2) - 0.01 * m.entropy()

                ratio_post = ratio
                surr1_post = ratio_post * advantage_post
                surr2_post = torch.clamp(ratio_post, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_post
                policy_loss_post = -torch.min(surr1_post, surr2_post) - 0.01 * m.entropy()

                self.optimizer_policy.zero_grad()
                (policy_loss.pow(2).mean() + policy_loss_post.pow(2).mean() + 0.5 * value.pow(2).mean() + 0.5 * value_post.pow(2).mean()).backward(retain_graph=True)
                self.optimizer_policy.step()
    
    def learn(self, n_episodes=1000, save_interval=100):
        # Train the agent
        for episode in range(n_episodes):
            state = self.env.reset()
            rewards = []
            rewards_pre_state = []
            rewards_post_state = []
            states = []
            next_states = []
            actions = []
            probs = []
            post_states = []
            # next_post_states = []
            done = False
            while not done:
                action, log_prob, prob, value, value_post = self.get_action(state)
                next_state, reward, done, info = self.env.step(action[0].detach().numpy())
                machine_setup, inventory_level, setup_cost = self.get_post_state(action[0].detach().numpy(), state[self.env.n_items:self.env.n_items+self.env.n_machines], state[0:self.env.n_items])
                post_state = state.copy()
                post_state[self.env.n_items:self.env.n_items+self.env.n_machines] = machine_setup
                post_state[0:self.env.n_items] = inventory_level
                post_states.append(post_state)
                post_state = torch.from_numpy(post_state).float().unsqueeze(0)
                rewards.append(reward)
                reward_pre_state = -(self.env.total_cost['holding_costs'] + self.env.total_cost['lost_sales'])
                reward_post_state = -setup_cost.sum()
                rewards_pre_state.append(reward_pre_state)
                rewards_post_state.append(reward_post_state)
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                probs.append(prob)
                
                state = next_state
                if done:
                    self.update(rewards, rewards_pre_state, rewards_post_state, states, post_states, actions, probs, next_states)
                    print('Episode:', episode, 'Reward:', sum(rewards))
                    if episode % save_interval == 0:
                        self.save(f'policy_{episode}.pt')            
            self.save(self.LOG_DIR)
    
    
    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'value_post_state_dict': self.value_post.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_value_state_dict': self.optimizer_value.state_dict(),
            'optimizer_value_post_state_dict': self.optimizer_value_post.state_dict()
        }, filepath)

    
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.value_post.load_state_dict(checkpoint['value_post_state_dict'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])
        self.optimizer_value_post.load_state_dict(checkpoint['optimizer_value_post_state_dict'])
    
    