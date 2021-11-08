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

        self.action = np.zeros(self.data_set.num_containers * 3)
        self.state = self.reset()
        self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.action.shape[0], ), dtype=np.int64)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state.flatten().shape[0], ), dtype=np.int64)

    def reset(self):
        y = self.data_set.system_manager.init_servers(self.data_set.system_manager._x)
        state = self.data_set.system_manager.get_next_state(y)
        return state

    def next_step(self, action):
        y = self.data_set.system_manager._y
        y = np.clip(y + action - 1, 0, 9)
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
    
    def get_y(self):
        return self.data_set.system_manager._y

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
        for d in self.data_set.system_manager.edge.values():
            E_d = d.energy_consumption()
            E_d_hat = d._energy
            energy_factor.append(E_d_hat / E_d)
        energy_factor = 0 # np.mean(energy_factor)

        reward = energy_factor + utility_factor
        #print("energy_factor", energy_factor)
        #print("utility_factor", utility_factor)
        for s in self.data_set.system_manager.server.values():
            if s.constraint_chk() == False:
                reward -= self.data_set.max_arrival
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
        for e in self.data_set.system_manager.edge.values():
            e.energy_update()

    def after_episode(self):
        for e in self.data_set.system_manager.edge.values():
            e._energy = 100.

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