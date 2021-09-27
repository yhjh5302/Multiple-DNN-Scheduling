
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
import pyro.distributions as dist

from model import (Actor, Critic)
from memory import SequentialMemory
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, env, num_states, num_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.env = env
        self.num_servers = self.env.data_set.num_servers
        self.num_containers = self.env.data_set.num_containers

        self.num_states = num_states
        self.num_actions = num_actions

        # Create Actor and Critic Network
        net_cfg = {'hidden1':args.hidden1, 'hidden2':args.hidden2, 'hidden3':args.hidden3, 'hidden4':args.hidden4, 'hidden5':args.hidden5, 'hidden6':args.hidden6, 'init_w':args.init_w}

        self.actor = Actor(self.num_states, self.num_containers, **net_cfg)
        self.actor_perturbed = Actor(self.num_states, self.num_containers, **net_cfg) # for parameter noise
        self.actor_target = Actor(self.num_states, self.num_containers, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.policy_learning_rate)

        self.critic = Critic(self.num_states, self.num_actions, **net_cfg)
        self.critic_target = Critic(self.num_states, self.num_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.learning_rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount

        # variables
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # delayed update for TD3
        self.iteration = 0
        self.policy_freq = args.policy_freq

        # cuda
        if USE_CUDA: self.cuda()

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_action_batch = self.actor_target(to_tensor(next_state_batch))
        next_action_batch, _, _ = self.env.action_batch_convert(next_action_batch, self.batch_size)
        target_q1_batch, target_q2_batch = self.critic_target([to_tensor(next_state_batch), to_tensor(next_action_batch)])
        target_q_batch = torch.min(target_q1_batch, target_q2_batch)
        target_q_batch = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(np.float)) * target_q_batch

        # Critic update
        self.critic.zero_grad()

        q1_batch, q2_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch)])

        value_loss = criterion(q1_batch, target_q_batch) + criterion(q2_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        policy_loss = 0
        if self.iteration % self.policy_freq == 0:
            self.actor.zero_grad()

            action_batch = self.actor(to_tensor(state_batch))
            action_batch, logprob, entropy = self.env.action_batch_convert(action_batch, self.batch_size)
            policy_loss = -self.critic.Q1([to_tensor(state_batch), to_tensor(action_batch)])
            q_batch = self.critic.Q1([to_tensor(state_batch), to_tensor(action_batch)])
            policy_loss = torch.mean(-logprob * q_batch) - torch.mean(entropy) * 0.001

            policy_loss.backward()
            self.actor_optim.step()

            # Target update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        self.iteration += 1
        return policy_loss

    def eval(self):
        self.actor.eval()
        self.actor_perturbed.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_perturbed.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def random_action(self, state):
        container_action = to_tensor(np.random.uniform(low=0.0, high=1.0, size=self.num_containers))
        server_action = to_tensor(np.random.uniform(low=0.0, high=1.0, size=self.num_containers))
        action = self.env.action_convert(container_action, server_action)

        self.a_t = action
        return action

    def select_action(self, state, ounoise=None, param_noise=None):
        if param_noise is None:
            container_alpha, container_beta, server_alpha, server_beta = self.actor(to_tensor(state))
        else:
            container_alpha, container_beta, server_alpha, server_beta = self.actor_perturbed(to_tensor(state))

        container_action = dist.Beta(container_alpha, container_beta).sample()
        server_action = dist.Beta(server_alpha, server_beta).sample()
        action = self.env.action_convert(container_action, server_action, ounoise)

        self.a_t = action
        return action

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            param += torch.randn(param.shape).cuda() * param_noise.current_betadev

    def reset(self, obs):
        self.s_t = obs

    def load_weights(self, output):
        if output is None:
            return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self,output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)