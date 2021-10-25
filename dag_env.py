import gym
from gym.utils import seeding
from svc_algorithm.dag_server import *
from svc_algorithm.dag_data_generator import *
from copy import deepcopy

import random
import torch
import torch.nn as nn
from pyro.distributions import Categorical, Beta
from util import *

import os
import multiprocessing as mp


class DAGEnv (gym.Env):
    def __init__(self, max_timeslot):
        self.data_set = DAGDataSet(max_timeslot=max_timeslot) # data gen

        self.cur_timeslot = 0
        self.max_timeslot = max_timeslot

        self.action = np.zeros(self.data_set.num_containers + self.data_set.num_servers)
        self.state = self.reset()
        self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.action.shape[0], ), dtype=np.int64)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state.shape[0], ), dtype=np.int64)

    def reset(self):
        y = self.data_set.system_manager.init_servers(self.data_set.system_manager._x)
        state = self.data_set.system_manager.get_next_state(y)
        return state

    def step(self, action):
        action = deepcopy(action)
        y = self.data_set.system_manager._y
        y[action[0]] = action[1]
        self.state = self.data_set.system_manager.get_next_state(y)
        reward = self.get_reward()

        done = False
        info = {}
        return self.state, reward, done, info

    def get_reward(self):
        T_n = self.data_set.system_manager.total_time()
        #U_n = self.calc_utility(T_n)
        #print("T_n", T_n)
        #print("U_n", U_n)

        utility_factor = 0
        total_arrival = np.sum(self.data_set.system_manager.service_arrival[self.cur_timeslot])
        for n in range(self.data_set.num_services):
            utility_factor += self.data_set.system_manager.service_arrival[self.cur_timeslot][n] / T_n[n] / total_arrival * self.data_set.max_arrival

        energy_factor = []
        for id, d in self.data_set.system_manager.edge.items():
            E_d = d.energy_consumption()
            E_d_hat = d._energy
            energy_factor.append(E_d_hat / E_d)
        energy_factor = 0 # np.mean(energy_factor)

        reward = energy_factor + utility_factor
        #print("energy_factor", energy_factor)
        #print("utility_factor", utility_factor)
        return reward

    def calc_utility(self, T_n):
        U_n = np.zeros(shape=(self.data_set.num_services, ))
        for n, svc in enumerate(self.data_set.svc_set.svc_set):
            T_n_hat = svc.deadline
            alpha = 2
            if T_n[n] < T_n_hat:
                U_n[n] = 1
            elif T_n_hat <= T_n[n] and T_n[n] < alpha * T_n_hat:
                U_n[n] = 1 - (T_n[n] - T_n_hat) / ((alpha - 1) * T_n_hat)
            else:
                U_n[n] = 0
        return U_n

    def after_timeslot(self):
        if self.cur_timeslot < self.max_timeslot - 1:
            self.cur_timeslot += 1
        else:
            self.cur_timeslot = 0
        self.data_set.system_manager.change_arrival(self.cur_timeslot)
        for id, e in self.data_set.system_manager.edge.items():
            e.energy_update()

    def after_episode(self):
        for id, e in self.data_set.system_manager.edge.items():
            e._energy = 100.

    def get_mask(self, c_id):
        cloud_id = self.data_set.system_manager.cloud_id
        system_manager = self.data_set.system_manager

        [system_manager.set_y(c_id, s_id) for s_id in system_manager.server]
        mask = [s.constraint_chk() for s_id, s in system_manager.server.items()]
        [system_manager.reset_y(c_id, s_id, cloud_id) for s_id in system_manager.server]

        mask[cloud_id] = -np.inf
        return np.array(mask)

    def action_convert(self, state, action, ounoise=None):
        if ounoise is not None:
            action += to_tensor(ounoise.noise())

        self.data_set.system_manager.set_y_mat(state[:self.data_set.num_containers])
        container_action = action[:self.data_set.num_containers]
        server_action = action[self.data_set.num_containers:]

        ''' for debug
        np.set_printoptions(precision=3, suppress=True)
        print("container_action", container_action)
        print("server_action", server_action)
        input()
        '''

        # get container id from action
        container_prob = nn.functional.softmax(container_action, dim=-1)
        container_dist = Categorical(container_prob)
        container_action = container_dist.sample()
        container_logprob = container_dist.log_prob(container_action)
        container_entropy = container_dist.entropy()

        # action masking
        server_mask = to_tensor(self.get_mask(container_action.item()))
        masked_server = torch.where(server_action > server_mask, server_mask, server_action)

        # get Y
        server_prob = nn.functional.softmax(masked_server, dim=-1)
        server_dist = Categorical(server_prob)
        server_action = server_dist.sample()
        server_logprob = server_dist.log_prob(server_action)
        server_entropy = server_dist.entropy()

        action = np.array([container_action.item(), server_action.item()])
        logprob = torch.stack([container_logprob, server_logprob], -1)
        entropy = torch.stack([container_entropy, server_entropy], -1)
        return action, logprob, entropy

    def action_batch_convert(self, state_batch, action_batch, batch_size):
        action = []
        logprob = []
        entropy = []

        #''' naive method
        y = self.data_set.system_manager._y

        for i in range(batch_size):
            act, log, ent = self.action_convert(state_batch[i], action_batch[i])
            action.append(act)
            logprob.append(log)
            entropy.append(ent)

        action = np.array(action)
        logprob = torch.stack(logprob)
        entropy = torch.stack(entropy)

        self.data_set.system_manager.set_y_mat(y)
        #'''
        
        ''' parallel method
        working_queue = [action for action in action_batch.cpu().detach()]
        with mp.Pool(processes=os.cpu_count() - 1) as pool:
            result = list(pool.map(self.action_convert, working_queue))
        result = np.array(result)
        print()
        '''

        return action, logprob, entropy

    def PrintState(self, state):
        start = 0
        end = start + self.data_set.num_containers
        print("y", state[start:end]) # y
        start = end
        end = start + self.data_set.num_containers
        print("x", state[start:end]) # x
        start = end
        end = start + self.data_set.num_containers
        print("container_arrival", state[start:end]) # container arrival
        start = end
        end = start + self.data_set.num_containers
        #print("container_computation_amount", state[start:end]) # container computation amount
        start = end
        end = start + self.data_set.num_containers
        #print("container_memory", state[start:end]) # container memory
        start = end
        end = start + self.data_set.num_servers
        print("server_cpu", state[start:end]) # server cpu, Tflops
        start = end
        end = start + self.data_set.num_servers
        print("server_memory", state[start:end]) # server memory, GBytes
        start = end
        end = start + self.data_set.num_servers
        print("server_energy", state[start:end]) # server energy