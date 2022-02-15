import gym
from dag_server import *
from dag_data_generator import *
from copy import deepcopy


class DAGEnv (gym.Env):
    def __init__(self, max_timeslot):
        self.data_set = DAGDataSet(max_timeslot=max_timeslot) # data gen
        self.scheduling_lst = np.array(sorted(zip(self.data_set.system_manager.ranku, np.arange(self.data_set.num_partitions)), reverse=True), dtype=np.int32)[:,1]

        self.cur_step = 0
        self.max_step = self.data_set.num_partitions
        self.cur_timeslot = 0
        self.max_timeslot = max_timeslot

        self.action = np.zeros(self.data_set.num_servers) # epsilon greedy placement (+greedy execution order(ranku)), 다른 방법으로 epsilon greedy execution order (+greedy placement)
        self.state = self.reset()
        self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.action.flatten().shape[0], ), dtype=np.int64)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state.flatten().shape[0], ), dtype=np.int64)

    def reset(self):
        self.cur_step = 0
        self.cur_timeslot = 0
        self.data_set.system_manager.init_env(execution_order=self.scheduling_lst)
        state = self.data_set.system_manager.get_state(next_p_id=self.scheduling_lst[0])
        return state

    def step(self, action):
        self.data_set.system_manager.set_env(cur_p_id=self.scheduling_lst[self.cur_step], s_id=action)
        reward = self.data_set.system_manager.get_reward(cur_p_id=self.scheduling_lst[self.cur_step], timeslot=self.cur_timeslot, step=self.cur_step)

        self.cur_step += 1
        if self.cur_step < self.max_step:
            done = False
        else:
            done = True
            self.cur_step = 0
        state = self.data_set.system_manager.get_state(next_p_id=self.scheduling_lst[self.cur_step])
        info = {}
        return state, reward, done, info

    def get_mask(self, step):
        mask = np.zeros(self.data_set.num_servers)
        p_id = self.scheduling_lst[step]
        for s_id in range(self.data_set.num_servers):
            self.data_set.system_manager.deployed_server[p_id] = s_id
            if self.data_set.system_manager.constraint_chk(deployed_server=self.data_set.system_manager.deployed_server, s_id=s_id):
                mask[s_id] = 1
        if sum(mask[:-1]) != 0:
            mask[-1] = 0
        return np.where(mask, 0, -np.inf)

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