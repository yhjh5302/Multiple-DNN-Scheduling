from matplotlib.pyplot import fill
from numpy import dtype
from algorithms.SAC.util import *
from algorithms.SAC.sac import SAC
from dag_server import *
from dag_data_generator import *


class DAGEnv():
    def __init__(self, dataset, layer_schedule, max_episode):
        self.dataset = dataset
        self.num_layers = len(layer_schedule)
        self.num_partitions = self.dataset.num_partitions
        self.num_requests = self.dataset.num_requests
        self.server_lst = self.dataset.server_lst
        self.num_servers = len(self.server_lst)
        self.deployed_server = np.full(self.num_partitions, fill_value=self.dataset.system_manager.cloud_id, dtype=np.int32)
        self.execution_order = self.dataset.system_manager.rank_u_schedule

        self.layer_schedule = layer_schedule

        self.cur_episode = 0
        self.max_episode = max_episode
        self.cur_timeslot = 0
        self.max_timeslot = dataset.num_timeslots
        self.cur_step = 0
        self.max_step = len(layer_schedule)

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
        x = (x + 1) / 2
        index = np.where(self.dataset.partition_layer_map == self.layer_schedule[step])[0]
        if step >= 0:
            while True:
                # 바로 여기서 각 서버에 대한 percentage를 partition 배치로 전환
                # 구체적으로는 partition별 workload를 계산해서 0.1 0.1 0.8이면 제일 큰 순서대로 
                portion = x / sum(x) * len(index)
                floor = np.floor(x / sum(x) * len(index))
                difference = portion - floor
                for _ in range(len(index) - int(sum(floor))):
                    maxidx = np.argmax(difference)
                    difference[maxidx] = 0
                    floor[maxidx] += 1
                portion, s_id = floor, 0
                for total_id in index:
                    for s_id, p in enumerate(portion):
                        if p == 0:
                            continue
                        else:
                            break
                    portion[s_id] -= 1
                    self.deployed_server[total_id] = self.server_lst[s_id]

                self.dataset.system_manager.set_env(deployed_server=self.deployed_server, execution_order=self.execution_order)
                constraint_chk = self.dataset.system_manager.constraint_chk()
                for idx, portion in enumerate(x):
                    if portion == 0:
                        constraint_chk[self.server_lst[idx]] = True
                if False in constraint_chk:
                    for idx, chk in enumerate(constraint_chk):
                        if chk == False:
                            x[self.server_lst.index(idx)] = 0
                    continue
                else:
                    self.dataset.system_manager.total_time_dp()
                    reward = -max(self.dataset.system_manager.finish_time[index])
                    break
            # print(self.deployed_server[index], "reward", reward, "constraint_chk", constraint_chk)
        else:
            constraint_chk = [True]
            reward = 0

        if step + 1 < self.max_step:
            next_step = step + 1
        else:
            next_step = 0
        next_partitions = np.where(self.dataset.partition_layer_map==self.layer_schedule[next_step])[0]
        # for i in next_partitions:
        #     print(self.dataset.svc_set.partitions[i].layer_name)
        next_svc = self.dataset.partition_service_map[next_partitions[0]]
        state = []
        state.extend([self.dataset.system_manager.server[s_id].computing_intensity[next_svc] / self.dataset.system_manager.server[s_id].computing_frequency * (10**9) for s_id in self.server_lst]) # server computing power
        state.extend([sum(self.dataset.system_manager.server[s_id].deployed_partition_memory.values()) / self.dataset.system_manager.server[s_id].memory for s_id in self.server_lst]) # server memory
        state.extend([self.dataset.system_manager.server[s_id].energy_consumption() / self.dataset.system_manager.server[s_id].cur_energy for s_id in self.server_lst]) # server energy
        # state.extend([max(self.dataset.system_manager.finish_time[np.where(self.dataset.partition_layer_map==layer)]) if layer in self.layer_schedule[:next_step] else 0 for layer in range(self.num_layers)]) # finish time
        # print(np.array(state))
        # self.PrintState(np.array(state))
        return np.array(state), reward, constraint_chk

    def next_step(self, action):
        self.state, reward, constraint_chk = self.get_state(action, self.cur_step)

        if self.cur_step + 1 < self.max_step and False not in np.array(constraint_chk)[self.server_lst]:
            done = False
            self.cur_step += 1
        else:
            # print(self.cur_step, reward, constraint_chk)
            # input()
            done = True
            self.cur_step = 0
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
        print("1: server computing power", state[start:end])
        start = end
        end += self.num_servers
        print("2: server memory", state[start:end])
        start = end
        end += self.num_servers
        print("3: server energy", state[start:end])
        input()