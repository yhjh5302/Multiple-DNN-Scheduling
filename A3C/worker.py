import numpy as np
import torch
import torch.nn.functional as F 
import torch.multiprocessing as mp
import pyro
from pyro.distributions import Categorical

from A3C.models import ValueNetwork, PolicyNetwork
from parameter_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from util import *


class Worker(mp.Process):
    def __init__(self, id, env, gamma, global_value_network, global_policy_network, global_value_optimizer, global_policy_optimizer, global_episode, GLOBAL_MAX_EPISODE):
        super(Worker, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = "w%i" % id
        
        self.env = env
        self.env.seed(id)
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.num_servers = self.env.data_set.num_servers

        self.gamma = gamma
        self.local_policy_network = PolicyNetwork(self.obs_dim, self.num_servers).to(self.device)
        self.pertubed_local_policy_network = PolicyNetwork(self.obs_dim, self.num_servers).to(self.device)
        self.local_value_network = ValueNetwork(self.obs_dim, 1).to(self.device)

        self.global_policy_network = global_policy_network.to(self.device)
        self.global_policy_optimizer = global_policy_optimizer
        self.global_value_network = global_value_network.to(self.device)
        self.global_value_optimizer = global_value_optimizer
        self.global_episode = global_episode
        self.GLOBAL_MAX_EPISODE = GLOBAL_MAX_EPISODE

        # sync local networks with global networks
        self.sync_with_global()

    def perturb_policy_network_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.pertubed_local_policy_network, self.local_policy_network)
        params = self.pertubed_local_policy_network.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape).cuda() * param_noise.current_stddev
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)

        logits = self.pertubed_local_policy_network.forward(state)
        action, mask = self.env.action_convert(logits)

        return action, mask
    
    def compute_loss(self, trajectory):
        states = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions = np.array([sars[1] for sars in trajectory])
        masks = np.array([sars[2] for sars in trajectory])
        rewards = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[4] for sars in trajectory]).to(self.device)
        dones = torch.FloatTensor([sars[5] for sars in trajectory]).to(self.device)
        
        # compute value target
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device) * rewards[j:]) for j in range(rewards.size(0))]
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)

        # compute value loss
        values = self.local_value_network.forward(states)
        value_loss = F.mse_loss(values, value_targets.detach())

        # compute policy loss with entropy bonus
        logits = self.local_policy_network.forward(states)
        actions, logprob, entropy = self.env.action_batch_convert(masks, logits)

        # compute entropy bonus
        entropy = torch.sum(entropy)

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

    def sync_with_global(self):
        self.local_value_network.load_state_dict(self.global_value_network.state_dict())
        self.local_policy_network.load_state_dict(self.global_policy_network.state_dict())

    def run(self):
        state = self.env.reset()
        trajectory = [] # [[s, a, m, r, s', done], [], ...]
        episode_reward = 0
        step = 0

        param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.001, desired_action_stddev=0.3, adaptation_coefficient=(1.05))
        perturbed_actions = list()
        unperturbed_actions = list()
        
        while self.global_episode.value < self.GLOBAL_MAX_EPISODE:
            action, mask = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            trajectory.append([state, action, mask, reward, next_state, done])
            episode_reward += reward
            step += 1

            perturbed_actions.append(to_numpy(self.pertubed_local_policy_network(to_tensor(state))))
            unperturbed_actions.append(to_numpy(self.local_policy_network(to_tensor(state))))

            if done:
                with self.global_episode.get_lock():
                    self.global_episode.value += 1
                print(self.name + " | episode: "+ str(self.global_episode.value) + " " + str(episode_reward / step))
                print(self.env.data_set.system_manager._y)

                self.update_global(trajectory)
                self.sync_with_global()

                trajectory = []
                episode_reward = 0
                step = 0
                state = self.env.reset()

                # noise update
                ddpg_dist = ddpg_distance_metric(np.array(perturbed_actions), np.array(unperturbed_actions))
                param_noise.adapt(ddpg_dist)
                self.perturb_policy_network_parameters(param_noise)
                perturbed_actions.clear()
                unperturbed_actions.clear()
            else:
                state = next_state