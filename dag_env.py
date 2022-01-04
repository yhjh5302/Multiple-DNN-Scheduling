import gym
from gym.utils import seeding
from dag_server import *
from dag_data_generator import *
from copy import deepcopy

import os
import multiprocessing as mp


class DAGEnv (gym.Env):
    def __init__(self, max_timeslot):
        self.data_set = DAGDataSet(max_timeslot=max_timeslot) # data gen

        self.cur_timeslot = 0
        self.max_timeslot = max_timeslot

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
        new_y = np.clip(y + action - 1, 0, 8)
        self.state = self.data_set.system_manager.get_next_state(new_y)
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