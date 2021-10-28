import numpy as np
import torch
import torch.nn.functional as F 
import torch.multiprocessing as mp
import pyro
from pyro.distributions import Categorical

from A3C.models import ContainerPolicyNetwork, ServerPolicyNetwork, ValueNetwork
from util import *


class Worker(mp.Process):
    def __init__(self, id, env, gamma, global_container_policy_network, global_container_policy_optimizer, global_server_policy_network, global_server_policy_optimizer, global_value_network, global_value_optimizer, global_episode, GLOBAL_MAX_EPISODE):
        super(Worker, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = "w%i" % id
        
        self.env = env
        self.env.seed(id)

        self.obs_dim = self.env.data_set.system_manager.NUM_CHANNEL
        self.action_dim = self.env.data_set.num_servers

        self.num_servers = self.env.data_set.num_servers
        self.num_containers = self.env.data_set.num_containers

        self.gamma = gamma
        self.local_container_policy_network = ContainerPolicyNetwork(self.obs_dim, self.num_containers, self.num_servers).to(self.device)
        self.local_server_policy_network = ServerPolicyNetwork(self.obs_dim, self.num_containers, self.num_servers).to(self.device)
        self.local_value_network = ValueNetwork(self.obs_dim, self.num_containers, self.num_servers).to(self.device)

        self.global_container_policy_network = global_container_policy_network.to(self.device)
        self.global_container_policy_optimizer = global_container_policy_optimizer
        self.global_server_policy_network = global_server_policy_network.to(self.device)
        self.global_server_policy_optimizer = global_server_policy_optimizer
        self.global_value_network = global_value_network.to(self.device)
        self.global_value_optimizer = global_value_optimizer
        self.global_episode = global_episode
        self.GLOBAL_MAX_EPISODE = GLOBAL_MAX_EPISODE

        # sync local networks with global networks
        self.sync_with_global()
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)

        container_logits = self.local_container_policy_network.forward(state.reshape(1,*state.shape))
        container_action = self.env.container_action_convert(container_logits)
        server_logits = self.local_server_policy_network.forward(state.reshape(1,*state.shape), container_action)
        server_action = self.env.server_action_convert(server_logits, container_action)

        one_hot_action = to_numpy(torch.cat([container_action.flatten(), server_action.flatten()]))
        action = np.array([container_action.flatten().tolist().index(1), server_action.flatten().tolist().index(1)])
        return one_hot_action, action
    
    def compute_loss(self, trajectory):
        ys = np.array([sars[0] for sars in trajectory])
        states = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
        actions = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[5] for sars in trajectory]).to(self.device)
        
        # compute value target
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device) * rewards[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        # compute value loss
        values = self.local_value_network.forward(states, actions)
        value_loss = F.mse_loss(values, value_targets.detach())

        # 
        container_logits = self.local_container_policy_network.forward(states)
        container_actions, container_logprob, container_entropy = self.env.container_action_batch_convert(ys, container_logits)
        server_logits = self.local_server_policy_network.forward(states, container_actions)
        server_actions, server_logprob, server_entropy = self.env.server_action_batch_convert(ys, server_logits, container_actions)

        # compute container policy loss & entropy bonus
        container_entropy = torch.sum(container_entropy)
        advantage = value_targets - values
        container_policy_loss = -container_logprob * advantage.detach()
        container_policy_loss = container_policy_loss.mean() - container_entropy * 0.001

        # compute server policy loss & entropy bonus
        server_entropy = torch.sum(server_entropy)
        advantage = value_targets - values
        server_policy_loss = -server_logprob * advantage.detach()
        server_policy_loss = server_policy_loss.mean() - server_entropy * 0.001
        
        return value_loss, container_policy_loss, server_policy_loss

    def update_global(self, trajectory):
        value_loss, container_policy_loss, server_policy_loss = self.compute_loss(trajectory)
        
        self.global_value_optimizer.zero_grad()
        value_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_value_network.parameters(), self.global_value_network.parameters()):
            global_params._grad = local_params._grad
        self.global_value_optimizer.step()
        
        self.global_container_policy_optimizer.zero_grad()
        container_policy_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_container_policy_network.parameters(), self.global_container_policy_network.parameters()):
            global_params._grad = local_params._grad
        self.global_container_policy_optimizer.step()
        
        self.global_server_policy_optimizer.zero_grad()
        server_policy_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_server_policy_network.parameters(), self.global_server_policy_network.parameters()):
            global_params._grad = local_params._grad
        self.global_server_policy_optimizer.step()

        return value_loss, container_policy_loss, server_policy_loss

    def sync_with_global(self):
        self.local_container_policy_network.load_state_dict(self.global_container_policy_network.state_dict())
        self.local_server_policy_network.load_state_dict(self.global_server_policy_network.state_dict())
        self.local_value_network.load_state_dict(self.global_value_network.state_dict())

    def run(self):

        self.env.data_set.system_manager.set_y_mat(np.array([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7], dtype=int))
        next_state, reward, done, _ = self.env.next_step(np.array([0, 8], dtype=int))
        print("max reward:", reward)

        state = self.env.reset()
        trajectory = [] # [[y, s, a, r, s', done], [], ...]
        episode_reward = 0
        step = 0
        
        while self.global_episode.value < self.GLOBAL_MAX_EPISODE:
            one_hot_action, action = self.get_action(state)
            next_state, reward, done, _ = self.env.next_step(action)
            trajectory.append([self.env.get_y(), state, one_hot_action, reward, next_state, done])
            episode_reward += reward
            step += 1
            #print("y", [[i] if idx == action[0] else i for idx, i in enumerate(self.env.get_y())])

            if done:
                with self.global_episode.get_lock():
                    self.global_episode.value += 1

                value_loss, container_policy_loss, server_policy_loss = self.update_global(trajectory)
                self.sync_with_global()

                print(self.name + " | #{} episode - avg_reward: {:.3f}, value_loss: {:.3f}, container_policy_loss: {:.3f}, server_policy_loss: {:.3f}".format(self.global_episode.value, episode_reward / step, value_loss, container_policy_loss, server_policy_loss))
                print("Y:", self.env.get_y(), "reward:", reward)
                #self.env.PrintState(state)

                trajectory = []
                episode_reward = 0
                step = 0
                state = self.env.reset()
            else:
                state = next_state