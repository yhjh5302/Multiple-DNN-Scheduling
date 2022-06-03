from SAC.util import *
from SAC.sac import SAC
from dag_server_ra import *
from dag_data_generator_ra import *


class DAGEnv():
    def __init__(self, dataset, max_episode):
        self.dataset = dataset
        self.graph = [np.unique(cg) for cg in dataset.coarsened_graph]
        self.num_partitions = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])
        self.num_requests = self.dataset.num_requests
        self.num_servers = self.dataset.num_locals + self.dataset.num_edges
        self.deployed_server = np.copy(self.dataset.partition_device_map)
        self.allocated_resources = np.ones(self.num_partitions, dtype=np.float_)

        self.cur_episode = 0
        self.max_episode = max_episode
        self.cur_timeslot = 0
        self.max_timeslot = dataset.num_timeslots
        self.cur_step = 0
        self.max_step = self.num_partitions * 3
        self.cur_partition = 0
        self.max_partition = len(dataset.partition_device_map)
        self.cur_svc = dataset.partition_device_map[self.cur_step]
        self.max_svc = dataset.num_services

        self.action = np.zeros(2)
        self.state = self.reset()

        self.agent = SAC(dataset=dataset, num_states=len(self.state), num_actions=len(self.action), env=self)

    def reset(self):
        self.cur_timeslot = 0
        self.cur_step = 0
        self.cur_svc = self.dataset.partition_device_map[self.cur_step]
        self.deployed_server = np.copy(self.dataset.partition_device_map)
        self.allocated_resources = np.ones(self.num_partitions, dtype=np.float_)
        self.dataset.system_manager.init_env()
        return self.get_state(0, np.ones(self.num_servers), p_id=-1, step=-1)[0]
    
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

    def get_state(self, x, y, p_id, step):
        if p_id >= 0:
            self.deployed_server[p_id] = self.dataset.num_requests + x
            self.allocated_resources[p_id] = (y[x] + 1) / 2
        temp_x = self.get_uncoarsened_action(self.deployed_server, dtype=np.int32)
        temp_y = self.get_uncoarsened_action(self.resource_convert(self.deployed_server, self.allocated_resources), dtype=np.float_)
        constraint_chk = self.dataset.system_manager.constraint_chk(deployed_server=temp_x, allocated_resources=temp_y)

        if step + 1 < self.max_partition:
            reward = max(-sum(self.dataset.system_manager.total_time()), -10) / 10 + 1 # 0
            next_p_id = p_id + 1
        else:
            reward = max(-sum(self.dataset.system_manager.total_time()), -10) / 10 + 1
            next_p_id = p_id + 1
            if p_id + 1 == self.max_partition:
                next_p_id = 0
        state = []
        state.extend([sum(self.allocated_resources[np.where(self.deployed_server==s.id)]) for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # remain resource
        state.extend([sum(s.deployed_partition_memory.values()) / s.memory for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # server memory
        state.extend([self.dataset.partition_workload[next_p_id] * s.computing_intensity[self.cur_svc] / s.computing_frequency for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # partition computation time
        state.extend([self.dataset.partition_memory[next_p_id] / s.memory for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # partition memory
        for s in self.deployed_server:
            state.extend([1 if idx == s - self.num_requests else 0 for idx in range(self.num_servers)])
        state.extend([0 if self.deployed_server[idx] < self.num_requests else r for idx, r in enumerate(self.resource_convert(self.deployed_server, self.allocated_resources))])
        state.extend([1 if idx == next_p_id else 0 for idx in range(self.max_partition)])
        # print("step", step, "p_id", p_id, "next_p_id", next_p_id, "reward", 10 - reward * 10)
        # self.PrintState(np.array(state))
        return np.array(state), reward, constraint_chk

    def next_step(self, action):
        x = action[0] # discrete
        y = action[1] # continuous
        self.state, reward, constraint_chk = self.get_state(x, y, self.cur_partition, self.cur_step)

        if self.cur_partition + 1 < self.max_partition:
            self.cur_partition += 1
        else:
            self.cur_partition = 0
        if self.cur_step + 1 < self.max_step and False not in constraint_chk[self.num_requests:self.num_requests+self.num_servers]:
            done = False
            self.cur_step += 1
            if self.cur_svc != self.dataset.partition_device_map[self.cur_partition]:
                self.cur_svc = self.dataset.partition_device_map[self.cur_partition]
        else:
            done = True
            self.cur_step = 0
            self.cur_partition = 0
            self.cur_svc = self.dataset.partition_device_map[self.cur_partition]
            print(self.deployed_server)
            print(self.resource_convert(self.deployed_server, self.allocated_resources))
            print(10 - reward * 10)
        info = {}
        return self.state, reward, done, info

    def get_reward(self):
        return self.dataset.system_manager.get_reward()

    def after_timeslot(self, deployed_server, allocated_resources):
        self.dataset.system_manager.after_timeslot(deployed_server=deployed_server, allocated_resources=allocated_resources, timeslot=self.cur_timeslot)

    def after_episode(self):
        self.reset()

    def PrintState(self, state):
        start = end = 0
        end += self.num_servers
        print("1: server resource", state[start:end])
        start = end
        end += self.num_servers
        print("2: server memory", state[start:end])
        start = end
        end += self.num_servers
        print("3: partition computation time", state[start:end])
        start = end
        end += self.num_servers
        print("4: partition memory", state[start:end])
        for idx, p in enumerate(range(self.num_partitions)):
            start = end
            end += self.num_servers
            # print("{}: partition deployed server".format(5+idx), state[start:end])
        start = end
        end += self.num_partitions
        print("{}: partition resource allocation(converted)".format(5+self.num_partitions), state[start:end])
        start = end
        end += self.num_partitions
        print("{}: partition resource allocation(original)".format(6+self.num_partitions), state[start:end])
        start = end
        end += self.num_partitions
        print("{}: current deployed partition".format(7+self.num_partitions), state[start:end])
        input()