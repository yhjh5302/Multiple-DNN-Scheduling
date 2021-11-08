import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.multiprocessing as mp
from pyro.distributions import Categorical, OneHotCategorical

from A3C.models import PolicyNetwork, ValueNetwork
from util import *
import random


class Worker(mp.Process):
    def __init__(self, id, env, gamma, global_policy_network, global_policy_optimizer, global_value_network, global_value_optimizer, global_episode, GLOBAL_MAX_EPISODE):
        super(Worker, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = "w%i" % id
        
        self.env = env
        self.env.seed(id)

        self.num_servers = self.env.data_set.num_servers
        self.num_containers = self.env.data_set.num_containers

        self.obs_dim = self.env.data_set.system_manager.NUM_CHANNEL
        self.action_dim = self.num_containers * 3 # move left, stay, move right

        self.gamma = gamma
        self.local_policy_network = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.local_value_network = ValueNetwork(self.obs_dim).to(self.device)

        self.global_policy_network = global_policy_network.to(self.device)
        self.global_policy_optimizer = global_policy_optimizer
        self.global_value_network = global_value_network.to(self.device)
        self.global_value_optimizer = global_value_optimizer
        self.global_episode = global_episode
        self.GLOBAL_MAX_EPISODE = GLOBAL_MAX_EPISODE

        # sync local networks with global networks
        self.sync_with_global()
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)

        logits = self.local_policy_network.forward(state.reshape(1,*state.shape))
        probs = nn.functional.softmax(logits.reshape(self.num_containers, 3), dim=-1)
        dists = Categorical(probs)
        actions = dists.sample()
        return to_numpy(actions)
    
    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
        rewards = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)
        
        # compute value target
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device) * rewards[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        # compute value loss
        values = self.local_value_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())

        # compute policy loss & entropy bonus
        logits = self.local_policy_network.forward(states)
        dists = nn.functional.softmax(logits.reshape(-1, self.num_containers, 3), dim=-1)
        probs = Categorical(dists)
        
        entropy = probs.entropy()
        entropy = torch.sum(entropy)

        logprob = probs.log_prob(actions)
        advantage = value_targets - values
        policy_loss = -logprob * advantage.detach()
        policy_loss = policy_loss.mean() - entropy * 0.001
        
        return value_loss, policy_loss

    def update_global(self, trajectory):
        value_loss, policy_loss = self.compute_loss(trajectory)
        
        self.global_value_optimizer.zero_grad()
        value_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_value_network.parameters(), self.global_value_network.parameters()):
            global_params._grad = local_params._grad
        self.global_value_optimizer.step()
        
        self.global_policy_optimizer.zero_grad()
        policy_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_policy_network.parameters(), self.global_policy_network.parameters()):
            global_params._grad = local_params._grad
        self.global_policy_optimizer.step()

        return value_loss, policy_loss

    def sync_with_global(self):
        self.local_policy_network.load_state_dict(self.global_policy_network.state_dict())
        self.local_value_network.load_state_dict(self.global_value_network.state_dict())

    def run(self):
        state = self.env.reset()
        trajectory = [] # [[y, s, a, r, s', done], [], ...]
        episode_reward = 0
        step = 0
        
        while self.global_episode.value < self.GLOBAL_MAX_EPISODE:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.next_step(action)
            trajectory.append([state, action, reward, next_state, done])
            episode_reward += reward
            step += 1
            #print("y", [[i] if idx == action[0] else i for idx, i in enumerate(self.env.get_y())])

            if done:
                with self.global_episode.get_lock():
                    self.global_episode.value += 1

                value_loss, policy_loss = self.update_global(trajectory)
                self.sync_with_global()
                trajectory = []

                print(self.name + " | #{} episode - avg_reward: {:.3f}".format(self.global_episode.value, episode_reward / self.env.max_step))
                print("Y:", self.env.get_y(), "reward:", reward, [s.constraint_chk() for s in self.env.data_set.system_manager.server.values()])
                #self.env.PrintState(state)

                episode_reward = 0
                state = self.env.reset()
            else:
                state = next_state