from networks_dqn import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import numpy as np

import os
from episode_memory import Memory
import random


CHECKPOINT= os.path.join(os.path.dirname(__file__), 'models_dqn/policy') # .pt'

def update_target_model(net, target_net):
    target_net.load_state_dict(net.state_dict())

class DQN:
    def __init__(self,  dim_obs, dim_act, train = True):
        #####################################
        self.replay_size = 100000
        self.batch_size = 256
        self.discount_factor = 0.99
        self.train = train
        if self.train:
            self.epsilon = 0.95
        else:
            self.epsilon = 0 # Greedy choice
        
        self.final_epsilon = 0.00  # Final epsilon value
        self.dec_epsilon = 0.00001  # Decrease rate of epsilon for every generation
        #####################################

        self.net = Agent(dim_obs, dim_act)  # Q-network
        self.test_net = Agent(dim_obs, dim_act)
        self.memory = Memory(self.replay_size)  # replay buffer                 # Replay buffer
        self.target_net = deepcopy(self.net)                               # Target Q-network
        self.update_steps = 100  # Update Target Network

        self._iterations = 0
        self.observation_steps = 300 # Number of iterations to observe before training every generation
        self.save_num = 100 # Save checkpoint
        self.num_inputs = dim_obs
        self.act_size = dim_act
        # self.target_net = Agent(self.num_inputs, self.act_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_net.load_state_dict(torch.load(CHECKPOINT + '.pt', map_location=torch.device(self.device)))
        
        # update_target_model(self.net, self.target_net)
        self.net.train()
        self.target_net.train()
        self.net.to(self.device)
        self.target_net.to(self.device)
        self.grad_norm_clip = 10
        self.loss = 0
        self.lr = 0.00005
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
    def soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - 0.95)*param.data + 0.95*target_param.data)
            
    def update_policy(self):
        batch = self.memory.sample(self.batch_size)
        states = torch.Tensor(batch.state).to(self.device)
        next_states = torch.Tensor(batch.next_state).to(self.device)
        actions = torch.Tensor(batch.action).long().to(self.device)
        rewards = torch.Tensor(batch.reward).to(self.device)
        dones = torch.Tensor(batch.done).to(self.device)

        q_values = self.net(states).squeeze(1)
    
        max_next_q_values = self.target_net(next_states).squeeze(1).max(1)[0]
        one_hot_action = torch.zeros(self.batch_size, q_values.size(-1)).to(self.device)
        one_hot_action.scatter_(1, actions.unsqueeze(1), 1)
        chosen_q_values = torch.sum(q_values.mul(one_hot_action), dim=1)
        target = rewards + self.discount_factor * max_next_q_values*(1-dones)
        target = rewards + self.discount_factor * max_next_q_values


        td_error = (chosen_q_values - target.detach())
        loss = ((td_error) ** 2).sum() 
        self.loss = loss.cpu().data.numpy()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        
        if self._iterations  % self.update_steps == 0: 
            update_target_model(self.net, self.target_net)
     
    ######################################################################################
    def select_action(self, state,test = False):
        if test:
            state = torch.Tensor(state)
            qvalue = self.test_net(state)
            qvalue = qvalue.cpu().data.numpy()
        else:
            state = torch.Tensor(state).to(self.device)    
            qvalue = self.net(state)
            qvalue = qvalue.cpu().data.numpy()
        if test:
            picked_actions = np.argmax(qvalue)
        else:
            pick_random = int(np.random.rand() <= self.epsilon)
            random_actions = random.randrange(self.act_size)
            picked_actions = pick_random * random_actions + (1 - pick_random) * np.argmax(qvalue)
        return picked_actions

    def store_experience(self, state, next_state, act, rew, done):
        self.memory.push(state, next_state, act, rew, done)
        self.epsilon = max(self.epsilon - self.dec_epsilon, self.final_epsilon)

    def save_checkpoint(self, iteration):
        if iteration % self.save_num ==0:
            self.net.save_model(self.net, CHECKPOINT+'.pt')

    def update(self):
        if len(self.memory) > self.observation_steps:
            self._iterations += 1
            self.update_policy()
            self.soft_update_target_network(self.target_net,self.net)
            
    def Epsilon(self):
        print("epsilon:",self.epsilon)
        
    def Loss(self):
        print("loss:",self.loss)