import gym
from gym.utils import seeding
from svc_algorithm.dag_server import *
from svc_algorithm.dag_data_generator import *
from copy import deepcopy

import random
import torch
import torch.nn as nn
import pyro.distributions as dist
from util import *

import os
import multiprocessing as mp


class DAGEnv (gym.Env):
    def __init__(self):
        mp.set_start_method('spawn')
        self.data_set = DAGDataSet() # data gen

        self.cur_timeslot = 0
        self.max_timeslot = 24

        self.action = self.data_set.system_manager.init_servers(self.data_set.system_manager._x)
        self.state = self.reset()
        self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.action.shape[0], ), dtype=np.int64)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state.shape[0], ), dtype=np.int64)

    def reset(self):
        action = self.data_set.system_manager.init_servers(self.data_set.system_manager._x)
        state = self.data_set.system_manager.get_next_state(action)
        state = self.Normalize(state)
        return state

    def step(self, action):
        action = deepcopy(action)
        self.state = self.data_set.system_manager.get_next_state(action)
        self.state = self.Normalize(self.state)
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

    def container_deploy(self, c_id, server):
        cloud_id = self.data_set.system_manager.cloud_id
        system_manager = self.data_set.system_manager
        server_list = (np.arange(self.data_set.num_servers) / self.data_set.num_servers)

        mask = self.get_mask(c_id)
        if True not in np.isposinf(mask):
            mask[cloud_id] = np.inf
        masked_server_list = np.where(server_list > mask, mask, server_list)
        
        s_id = np.argmin(pow(masked_server_list - server, 2))
        system_manager.set_y(c_id, s_id)
        return s_id

    def action_convert(self, container_action, server_action, ounoise=None):
        container_action = to_numpy(container_action)
        server_action = to_numpy(server_action)

        if ounoise is not None:
            container_action += ounoise.noise()
            server_action += ounoise.noise()

        ''' for debug
        np.set_printoptions(precision=3, suppress=True)
        print("container_action", container_action)
        print("server_action", server_action)
        '''

        action = np.zeros(self.data_set.num_containers)
        self.reset()
        for _ in range(self.data_set.num_containers):
            c_id = np.argmax(container_action)
            action[c_id] = self.container_deploy(c_id, server_action[c_id])
            # after loop
            container_action[c_id] = -np.inf

        return action

    def action_batch_convert(self, action_batch, batch_size):
        container_alpha, container_beta, server_alpha, server_beta = action_batch

        action = []
        logprob = []
        entropy = []
        for i in range(batch_size):
            container_dist = dist.Beta(container_alpha[i], container_beta[i])
            container_action = container_dist.sample()
            container_logprob = container_dist.log_prob(container_action)
            container_entropy = container_dist.entropy()

            server_dist = dist.Beta(server_alpha[i], server_beta[i])
            server_action = server_dist.sample()
            server_logprob = server_dist.log_prob(server_action)
            server_entropy = server_dist.entropy()

            action.append(self.action_convert(container_action, server_action))
            logprob.append(torch.cat([container_logprob, server_logprob], -1))
            entropy.append(torch.cat([container_entropy, server_entropy], -1))

        action = np.array(action)
        logprob = torch.stack(logprob)
        entropy = torch.stack(entropy)
        
        ''' parallel method
        working_queue = [action for action in action_batch.cpu().detach()]
        with mp.Pool(processes=os.cpu_count() - 1) as pool:
            result = list(pool.map(self.action_convert, working_queue))
        result = np.array(result)
        '''

        return action, logprob, entropy

    def Normalize(self, state):
        start = 0
        end = start + self.data_set.num_containers
        container_arrival = state[start:end] = state[start:end] # container arrival
        start = end
        end = start + self.data_set.num_containers
        container_computation_amount = state[start:end] = state[start:end] / (10**12) # container computation amount, Tflop
        start = end
        end = start + self.data_set.num_containers
        container_memory = state[start:end] = state[start:end] / (1024 * 1024)  # container memory, GBytes
        start = end
        end = start + self.data_set.num_servers
        server_cpu = state[start:end] = state[start:end] / (10**12) # server cpu, Tflops
        start = end
        end = start + self.data_set.num_servers
        server_memory = state[start:end] = state[start:end] / (1024 * 1024) # server memory, GBytes
        start = end
        end = start + self.data_set.num_servers
        server_energy = state[start:end] = state[start:end] / 100 # server energy
        return state

    def PrintState(self, state):
        start = 0
        end = start + self.data_set.num_containers
        #print("container_arrival", state[start:end]) # container arrival
        start = end
        end = start + self.data_set.num_containers
        #print("container_computation_amount", state[start:end]) # container computation amount
        start = end
        end = start + self.data_set.num_containers
        #print("container_memory", state[start:end]) # container memory
        start = end
        end = start + self.data_set.num_servers
        #print("server_cpu", state[start:end]) # server cpu, Tflops
        start = end
        end = start + self.data_set.num_servers
        #print("server_memory", state[start:end]) # server memory, GBytes
        start = end
        end = start + self.data_set.num_servers
        #print("server_energy", state[start:end]) # server energy