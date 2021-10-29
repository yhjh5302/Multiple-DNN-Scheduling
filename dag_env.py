import gym
from gym.utils import seeding
from dag_server import *
from dag_data_generator import *
from copy import deepcopy

import random
import torch
import torch.nn as nn
from pyro.distributions import OneHotCategorical, Categorical, Beta
from util import *

import os
import multiprocessing as mp


class DAGEnv (gym.Env):
    def __init__(self, max_timeslot, max_step):
        self.data_set = DAGDataSet(max_timeslot=max_timeslot) # data gen

        self.cur_timeslot = 0
        self.max_timeslot = max_timeslot
        self.cur_step = 0
        self.max_step = max_step

        self.action = np.zeros(self.data_set.num_servers)
        self.state = self.reset()
        self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.action.shape[0], ), dtype=np.int64)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state.flatten().shape[0], ), dtype=np.int64)

    def reset(self):
        y = self.data_set.system_manager.init_servers(self.data_set.system_manager._x)
        state = self.data_set.system_manager.get_next_state(y)
        return state

    def next_step(self, c_id, action):
        y = self.data_set.system_manager._y
        y[c_id] = action
        self.state = self.data_set.system_manager.get_next_state(y)
        reward = self.get_reward()

        self.cur_step += 1
        if self.cur_step < self.max_step:
            done = False
        else:
            done = True
            self.cur_step = 0
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
    
    def get_current_c_id(self, state):
        return np.where(state[0,self.data_set.system_manager.cloud_id,:] == 1)[0][0]

    def action_convert(self, c_id, action):
        # action masking
        mask = to_tensor(self.get_mask(c_id))
        masked_action = torch.where(action > mask, mask, action)

        # get Y
        prob = nn.functional.softmax(masked_action, dim=-1)
        dist = OneHotCategorical(prob)
        action = dist.sample()
        return action, to_numpy(mask)

    def action_batch_convert(self, mask_batch, action_batch):
        logprob = []
        entropy = []

        for i in range(action_batch.shape[0]):
            mask = to_tensor(mask_batch[i])
            masked_action = torch.where(action_batch[i] > mask, mask, action_batch[i])
            prob = nn.functional.softmax(masked_action, dim=-1)
            dist = OneHotCategorical(prob)
            action = dist.sample()
            logprob.append(dist.log_prob(action))
            entropy.append(dist.entropy())

        logprob = torch.stack(logprob)
        entropy = torch.stack(entropy)
        return logprob, entropy

    def PrintState(self, state):
        print("0: which container which server", state[:,0,:,:])
        print("1: deployed container computation amount", state[:,1,:,:])
        print("2: deployed container memory", state[:,2,:,:])
        print("3: deployed container arrival rate", state[:,3,:,:])
        print("4: server cpu", state[:,4,:,:])
        print("5: server memory", state[:,5,:,:])
        print("6: server energy", state[:,6,:,:])
        for c_id in range(self.data_set.num_containers):
            print("7+%d: dependency"%c_id, state[:,7+c_id,:,:])
        input()