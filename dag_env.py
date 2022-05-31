import torch
import torch.nn as nn
from pyro.distributions import OneHotCategorical, Categorical, Beta
from SAC.util import *
from SAC.sac import SAC
from dag_server_ra import *
from dag_data_generator_ra import *

import os
import argparse
import multiprocessing as mp


class DAGEnv():
    def __init__(self, dataset, max_episode):
        self.dataset = dataset
        self.graph = [np.unique(cg) for cg in dataset.coarsened_graph]
        self.num_partitions = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])
        self.deployed_server = np.copy(self.dataset.partition_device_map)
        self.allocated_resources = np.ones(self.num_partitions, dtype=np.float_)

        self.action = np.zeros(2)
        self.state = self.reset()

        self.cur_episode = 0
        self.max_episode = max_episode
        self.cur_timeslot = 0
        self.max_timeslot = dataset.num_timeslots
        self.cur_step = 0
        self.max_step = self.num_partitions
        self.cur_svc = dataset.partition_device_map[self.cur_step]
        self.max_svc = dataset.num_services

        self.agent = SAC(dataset=dataset, num_states=len(self.state), num_actions=len(self.action), env=self)

    def reset(self):
        self.cur_timeslot = 0
        self.cur_step = 0
        self.cur_svc = self.dataset.partition_device_map[self.cur_step]
        self.dataset.system_manager.init_env()
        return self.get_state(0, 1)
    
    def sample(self):
        return [random.uniform(0., 1.), random.randint(0, self.dataset.num_locals + self.dataset.num_edges - 1)]
    
    def get_uncoarsened_action(self, x, dtype):
        result = []
        start = end = 0
        for svc in self.dataset.svc_set.services:
            uncoarsened_x = np.zeros_like(self.dataset.coarsened_graph[svc.id], dtype=dtype)
            start = end
            end += len(self.graph[svc.id])
            for i, x_i in enumerate(x[start:end]):
                uncoarsened_x[np.where(self.dataset.coarsened_graph[svc.id]==self.graph[svc.id][i])] = x_i
            result.append(uncoarsened_x)
        return np.concatenate(result, axis=None)
    
    def resource_convert(self, x, y):
        temp = np.zeros_like(y)
        for s_id in np.unique(x):
            index = np.where(x == s_id)
            temp[index] = y[index] / sum(y[index])
        return temp

    def get_state(self, x, y):
        self.deployed_server[self.cur_step] = self.dataset.num_requests + x
        self.allocated_resources[self.cur_step] = y
        self.dataset.system_manager.set_env(deployed_server=self.get_uncoarsened_action(self.deployed_server, dtype=np.int32), allocated_resources=self.get_uncoarsened_action(self.resource_convert(self.deployed_server, self.allocated_resources), dtype=np.float_))

        state = []
        state.extend([sum([p.workload_size * s.computing_intensity[p.service.id] for p in s.deployed_partition.values()]) / s.computing_frequency for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # server computation info
        state.extend([max(s.deployed_partition_memory.values(), default=0) / s.memory for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # server memory
        state.extend([s.energy_consumption() / s.cur_energy for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # server energy
        return np.array(state)

    def next_step(self, action):
        x = action[1] # discrete
        y = (action[0] + 1) / 2 # continuous
        self.state = self.get_state(x, y)
        reward = self.get_reward()

        self.cur_step += 1
        if self.cur_step < self.max_step:
            self.cur_svc = self.dataset.partition_device_map[self.cur_step]
            done = False
            reward = 0
        else:
            done = True
            self.cur_step = 0
            self.cur_svc = self.dataset.partition_device_map[self.cur_step]
            reward += 1
            print(self.deployed_server)
            print(self.resource_convert(self.deployed_server, self.allocated_resources))
            print(reward - 1)
        info = {}
        return self.state, reward, done, info

    def get_reward(self):
        return self.dataset.system_manager.get_reward()

    def after_timeslot(self, deployed_server, allocated_resources):
        self.dataset.system_manager.after_timeslot(deployed_server=deployed_server, allocated_resources=allocated_resources, timeslot=self.cur_timeslot)

    def after_episode(self):
        self.reset()

    def train(self, validate_episodes, debug=False):
        self.agent.is_training = True
        for episode in range(self.max_episode):

            for timeslot in range(self.max_timeslot):
                observation = None

                for step in range(self.max_step):
                    # reset if it is the start of episode
                    if observation is None:
                        observation = deepcopy(self.reset())
                        self.agent.reset(observation)

                    # agent pick action ...
                    action = self.agent.select_action(state=observation, step=step)

                    # env response with next_observation, reward, terminate_info
                    observation2, reward, done, info = self.next_step(action)
                    observation2 = deepcopy(observation2)
                    if step >= self.max_step - 1:
                        done = True

                    # agent observe and update policy
                    self.agent.observe(reward, observation2, done)
                    self.agent.update_policy()

                    # update
                    observation = deepcopy(observation2)

                    if done:
                        break

                # after timeslot - update battery
                #env.after_timeslot()

                if debug:
                    prGreen('#{}: timeslot:{} avg_timeslot_reward:{}'.format(episode, timeslot, reward))

            # after episode - reset environment
            #env.after_episode()

            if debug:
                prCyan('#{}: avg_episode_reward:{}'.format(episode, reward))

            # [optional] save intermideate model
            if episode % int(self.max_episode / 5) == 0:
                self.agent.save_model('outputs')

            # [optional] evaluate
            if validate_episodes > 0 and episode % validate_episodes == 0:
                policy = lambda x, y: self.agent.select_action(x, y)
                validate_reward = self.test(policy, debug=debug)
                if debug:
                    prYellow('[Evaluate] Episode_{:04d}: mean_reward:{}'.format(episode, validate_reward))

    def test(self, policy, debug=False):
        self.is_training = False
        observation = self.reset()
        episode_reward = 0

        for timeslot in range(self.max_timeslot):
            for step in range(self.max_step):

                # reset at the start of timeslot

                step_reward = 0.
                assert observation is not None

                # start timeslot
                action = policy(observation)
                observation, reward, done, info = self.next_step(action)
                step_reward = reward

                if done: break

                # env.after_timeslot()
                # episode_reward += step_reward

                if debug:
                    prYellow('[Evaluate] #step{}: step_reward:{}'.format(step, step_reward))

                episode_reward = observation[0]
        self.after_episode()
        return 

    def get_mask(self, c_id):
        cloud_id = self.dataset.system_manager.cloud_id
        system_manager = self.dataset.system_manager

        [system_manager.set_y(c_id, s_id) for s_id in system_manager.server]
        mask = [s.constraint_chk() for s in system_manager.server.values()]
        [system_manager.reset_y(c_id, s_id, cloud_id) for s_id in system_manager.server]

        mask[cloud_id] = -np.inf
        return np.array(mask)

    def PrintState(self, state):
        print("0: which container which server", state[:,0,:,:])
        print("1: deployed container computation amount", state[:,1,:,:])
        print("2: deployed container memory", state[:,2,:,:])
        print("3: deployed container arrival rate", state[:,3,:,:])
        print("4: server cpu", state[:,4,:,:])
        print("5: server memory", state[:,5,:,:])
        print("6: server energy", state[:,6,:,:])
        for c_id in range(self.dataset.num_containers):
            print("7+%d: dependency"%c_id, state[:,7+c_id,:,:])
        input()