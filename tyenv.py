import gym
from gym.utils import seeding
from svc_algorithm.server import *
from svc_algorithm.data_generator import *


def Normalize(observation, num_server, num_container):
    start = 0
    end = num_container
    if np.sum(observation[start:end]) != 0:
        observation[start:end] = observation[start:end] / (10 ** 12) # X
    start = end
    end = start + num_container
    observation[start:end] = observation[start:end] # Y
    start = end
    end = start + num_server
    observation[start:end] = observation[start:end] / (10 ** 12) # remain cpu
    start = end
    end = start + num_server
    observation[start:end] = observation[start:end] / (1024 * 512) # remain mem
    start = end
    end = start + num_server
    observation[start:end] = observation[start:end] / 100 # remain energy
    return observation

class TYEnv (gym.Env):
    def __init__(self):
        self.data_set = None
        self.load_data_set()

        self.action = self.data_set.system_manager._x
        self.state = self.data_set.system_manager.get_state(self.action)
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.action.shape[0], ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state.shape[0], ), dtype=np.float32)

        self.cur_step = 0
        self.max_step = 200

    def load_data_set(self):
        self.data_set = DataSet() # data gen

    def reset(self):
        self.data_set.system_manager.set_y_mat(self.data_set.system_manager._y)
        self.action.fill(0)
        self.state = self.data_set.system_manager.get_state(self.action)
        return self.state

    def step(self, action):
        action = self.constraint_chk(action)
        self.state = self.data_set.system_manager.get_state(self.state[:action.shape[0]] + action)
        reward = self.observe()

        if self.cur_step >= self.max_step - 1:
            self.cur_step = 0
            done = True
        else:
            self.cur_step += 1
            done = False
        info = {}
        return self.state, reward, done, info

    def observe(self):
        self.data_set.system_manager.update_edge_computing_time()
        T_n = self.data_set.system_manager.total_time()
        U_n = self.calc_utility(T_n)

        utility_factor = 0
        for n in range(self.data_set.num_services):
            utility_factor += self.data_set.system_manager.service_arrival[n] * U_n[n]

        energy_factor = np.inf
        for d in self.data_set.system_manager.edge:
            E_d = d.energy_consumption()
            E_d_hat = d._energy
            energy_factor = min(energy_factor, E_d_hat / E_d)

        reward = energy_factor * utility_factor
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
    
    def constraint_chk(self, action):
        for i in range(self.action.shape[0]):
            if self.state[i] + action[i] < 0:
                action[i] = self.state[i] = 0

        temp_state = self.state[:self.action.shape[0]] + action
        self.data_set.system_manager.set_x_mat(temp_state)
        for e in self.data_set.system_manager.edge:
            if e.used_cpu > e.cpu:
                total_dec = total_inc = 0
                for c_id in e.deployed_container:
                    if action[c_id] < 0:
                        total_dec -= action[c_id]
                    else:
                        total_inc += action[c_id]
                for c_id in e.deployed_container:
                    if action[c_id] > 0:
                        action[c_id] *= total_dec / total_inc

        temp_state = self.state[:self.action.shape[0]] + action
        self.data_set.system_manager.set_x_mat(temp_state)
        for e in self.data_set.system_manager.edge:
            if e.used_mem > e.memory or e._energy < e.energy_consumption():
                for c_id in e.deployed_container:
                    action[c_id] = 0
        return action

    def after_timeslot(self, state):
        self.data_set.system_manager.set_x_mat(state[:self.action.shape[0]])
        for e in self.data_set.system_manager.edge:
            e.energy_update()

    def after_episode(self):
        for e in self.data_set.system_manager.edge:
            e._energy = 100.