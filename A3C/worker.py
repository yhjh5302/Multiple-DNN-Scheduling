import numpy as np
import torch
import torch.nn.functional as F 
import torch.multiprocessing as mp
import pyro
from pyro.distributions import Categorical

from A3C.models import Network
from util import *


class Worker(mp.Process):
    def __init__(self, id, env, gamma, global_network, global_optimizer, global_episode, GLOBAL_MAX_EPISODE):
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
        self.local_network = Network(self.obs_dim, self.num_containers, self.num_servers).to(self.device)

        self.global_network = global_network.to(self.device)
        self.global_optimizer = global_optimizer
        self.global_episode = global_episode
        self.GLOBAL_MAX_EPISODE = GLOBAL_MAX_EPISODE

        # sync local networks with global networks
        self.sync_with_global()
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)

        container_logits, server_logits = self.local_network.forward_logits(state.reshape(1,*state.shape))
        action = self.env.action_convert(container_logits, server_logits)

        return action
    
    def compute_loss(self, trajectory):
        ys = np.array([sars[0] for sars in trajectory])
        states = torch.FloatTensor([sars[1] for sars in trajectory]).to(self.device)
        actions = np.array([sars[2] for sars in trajectory])
        rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[5] for sars in trajectory]).to(self.device)
        
        # compute value target
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device) * rewards[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        # compute value loss
        container_logits, server_logits, values = self.local_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())
        logprob, entropy = self.env.action_batch_convert(ys, container_logits, server_logits)

        # compute entropy bonus
        entropy = torch.sum(entropy)

        # compute policy loss
        advantage = value_targets - values
        policy_loss = -logprob * advantage.detach()
        policy_loss = policy_loss.mean() - entropy * 0.001
        
        return value_loss, policy_loss

    def update_global(self, trajectory):
        value_loss, policy_loss = self.compute_loss(trajectory)
        
        self.global_optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_network.parameters(), self.global_network.parameters()):
            global_params._grad = local_params._grad
        self.global_optimizer.step()

        return value_loss, policy_loss

    def sync_with_global(self):
        self.local_network.load_state_dict(self.global_network.state_dict())

    def run(self):
        state = self.env.reset()
        trajectory = [] # [[y, s, a, r, s', done], [], ...]
        episode_reward = 0
        step = 0
        
        while self.global_episode.value < self.GLOBAL_MAX_EPISODE:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.next_step(action)
            trajectory.append([self.env.get_y(), state, action, reward, next_state, done])
            episode_reward += reward
            step += 1
            #print("y", [[i] if idx == action[0] else i for idx, i in enumerate(self.env.get_y())])

            if done:
                with self.global_episode.get_lock():
                    self.global_episode.value += 1

                value_loss, policy_loss = self.update_global(trajectory)
                self.sync_with_global()

                print(self.name + " | episode: "+ str(self.global_episode.value) + " " + str(episode_reward / step), "value loss:", float(value_loss), "policy_loss:", float(policy_loss))
                print("Y", self.env.get_y())
                #self.env.PrintState(state)

                trajectory = []
                episode_reward = 0
                step = 0
                state = self.env.reset()
            else:
                state = next_state