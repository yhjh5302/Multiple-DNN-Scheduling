from matplotlib.pyplot import fill
from numpy import dtype
from SAC.util import *
from SAC.sac import SAC
from dag_server import *
from dag_data_generator import *


class DAGEnv():
    def __init__(self, dataset, max_episode):
        self.dataset = dataset
        self.num_layers = len(np.unique(self.dataset.partition_layer_map))
        print("self.num_layers", self.num_layers)
        self.num_partitions = self.dataset.num_partitions
        self.num_requests = self.dataset.num_requests
        self.server_lst = self.dataset.server_lst
        self.num_servers = len(self.server_lst)
        self.deployed_server = np.full(self.num_partitions, fill_value=self.dataset.system_manager.cloud_id, dtype=np.int32)
        self.execution_order = self.dataset.system_manager.EFT_schedule

        self.cur_episode = 0
        self.max_episode = max_episode
        self.cur_timeslot = 0
        self.max_timeslot = dataset.num_timeslots
        self.cur_step = 0
        self.max_step = self.num_partitions

        self.action = np.zeros(self.num_servers)
        self.state = self.reset()[0]

        self.agent = SAC(num_states=len(self.state), num_actions=len(self.action), env=self, dataset=dataset)

    def reset(self):
        self.cur_timeslot = 0
        self.cur_step = 0
        self.deployed_server = np.full(self.num_partitions, fill_value=self.dataset.system_manager.cloud_id, dtype=np.int32)
        self.dataset.system_manager.init_env()
        return self.get_state(self.num_servers-1, step=-1)[0], np.zeros(self.num_servers)

    def get_state(self, x, step):
        if step >= 0:
            # 바로 여기서 각 서버에 대한 percentage를 partition 배치로 전환
            # 구체적으로는 partition별 workload를 계산해서 0.1 0.1 0.8이면 제일 큰 순서대로 
            index = self.dataset.partition_layer_map[step]
            for idx, ratio in enumerate(x):
                print("ratio", (ratio + 1) / 2)
            print("x", x)
            input()
            self.deployed_server[step] = self.server_lst[x]
        constraint_chk = self.dataset.system_manager.constraint_chk(deployed_server=self.deployed_server, execution_order=self.execution_order)

        if step + 1 < self.max_step:
            reward = 0
            next_step = step + 1
        else:
            reward = max(-max(self.dataset.system_manager.total_time()), -10) / 10 + 1
            next_step = 0
        state = []
        state.extend([sum(self.execution_order[np.where(self.deployed_server==s.id)]) for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # remain resource
        state.extend([sum(s.deployed_partition_memory.values()) / s.memory for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # server memory
        state.extend([self.dataset.partition_workload_map[next_step] * s.computing_intensity[self.dataset.partition_service_map[next_step]] / s.computing_frequency for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # partition computation time
        state.extend([self.dataset.partition_memory_map[next_step] / s.memory for s in list(self.dataset.system_manager.local.values())+list(self.dataset.system_manager.edge.values())]) # partition memory
        for s in self.deployed_server:
            state.extend([1 if idx == s - self.num_requests else 0 for idx in range(self.num_servers)])
        state.extend([1 if idx == next_step else 0 for idx in range(self.max_step)])
        # print("step", step, "p_id", p_id, "next_p_id", next_p_id, "reward", 10 - reward * 10)
        # self.PrintState(np.array(state))
        return np.array(state), reward, constraint_chk

    def next_step(self, action):
        self.state, reward, constraint_chk = self.get_state(action, self.cur_step)

        if self.cur_step + 1 < self.max_step and False not in np.array(constraint_chk)[self.server_lst]:
            done = False
            self.cur_step += 1
        else:
            done = True
            self.cur_step = 0
            print(self.deployed_server)
            print(10 - reward * 10)
        info = {}
        return self.state, reward, done, info

    def get_reward(self):
        return self.dataset.system_manager.get_reward()

    def after_timeslot(self, deployed_server, execution_order):
        self.dataset.system_manager.after_timeslot(deployed_server=deployed_server, execution_order=execution_order, timeslot=self.cur_timeslot)

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