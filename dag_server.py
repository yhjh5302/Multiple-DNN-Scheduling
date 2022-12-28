import time
import random
import math
import numpy as np
from copy import deepcopy

import dag_completion_time


COMP_RATIO = 50*(10**6) # computation per input data (50 MFLOPS)
MEM_RATIO = 1024 # memory usage per input data (1 KB)


class NetworkManager:  # managing data transfer
    def __init__(self, channel_bandwidth, gaussian_noise, B_edge_up, B_edge_down, B_cloud_up, B_cloud_down, request, local, edge, cloud):
        self.C = channel_bandwidth
        self.N_0 = gaussian_noise
        self.request_device = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.original_bandwidth = [B_edge_up, B_edge_down, B_cloud_up, B_cloud_down]
        self.B_edge_up = B_edge_up
        self.B_edge_down = B_edge_down
        self.B_cloud_up = B_cloud_up
        self.B_cloud_down = B_cloud_down
        self.B_dd = None
        self.P_d = None
        self.g_dd = None

        self.request = list(request.keys())
        self.local = list(local.keys())
        self.edge = list(edge.keys())
        self.cloud = list(cloud.keys())

        self.memory_bandwidth = 1024*1024*1024*20 # DDR4

    def communication(self, amount, sender, receiver):
        if sender == receiver:
            return amount / self.memory_bandwidth
        elif sender in self.cloud:
            return amount / self.B_cloud_down
        elif receiver in self.cloud:
            return amount / self.B_cloud_up
        elif sender in self.edge:
            return amount / self.B_edge_down
        elif receiver in self.edge:
            return amount / self.B_edge_up
        else:
            return amount / self.B_dd[sender, receiver]

    def cal_b_dd(self):
        self.B_dd = np.zeros_like(self.g_dd)
        for i in range(self.B_dd.shape[0]):
            for j in range(i + 1, self.B_dd.shape[1]):
                SINR = (self.g_dd[i,j] * self.P_d[i]) / (self.N_0)
                self.B_dd[i, j] = self.B_dd[j, i] = self.C * math.log2(1 + SINR)
            self.B_dd[i, i] = float("inf")

    def bandwidth_change(self, num_request=1):
        self.cal_b_dd()
        self.B_dd = self.B_dd / num_request


class SystemManager():
    def __init__(self):
        self.deployed_server = None
        self.execution_order = None
        self.cloud = None
        self.edge = None
        self.local = None
        self.request = None
        self.server = None
        self.computing_frequency = None
        self.computing_intensity = None
        self.computation_time_table = None

        # service and partition info
        self.service_set = None
        
        # for network bandwidth and arrival rate
        self.net_manager = None

        self.num_timeslots = None
        self.num_servers = None
        self.num_services = None
        self.num_partitions = None
        self.cloud_id = None
        self.timeslot = 0
        self.rank_u = None
        self.rank_d = None
        self.optimistic_cost_table = None

        self.scheduling_policy = 'rank_u' # 'rank_u', 'rank_d', 'rank_oct'
        self.rank_u_schedule = None
        self.rank_d_schedule = None
        self.rank_oct_schedule = None

    def calculate_schedule(self):
        num_partitions = len(self.service_set.partitions)

        # execution_order calculation, rank_u
        self.rank_u = np.zeros(num_partitions)
        for partition in self.service_set.partitions:
            if len(partition.predecessors) == 0:
                self.calc_rank_u_total_average(partition)
        self.rank_u_schedule = np.array(np.array(sorted(zip(self.rank_u, np.arange(num_partitions)), reverse=True), dtype=np.int32)[:,1], dtype=np.int32)
    
        # execution_order calculation, rank_d
        self.rank_d = np.zeros(num_partitions)
        for partition in self.service_set.partitions:
            if len(partition.successors) == 0:
                self.calc_rank_d_total_average(partition)
        self.rank_d_schedule = np.array(np.array(sorted(zip(self.rank_d, np.arange(num_partitions)), reverse=False), dtype=np.int32)[:,1], dtype=np.int32)

    def calculate_rank_oct_schedule(self, server_lst):
        # calculate optimistic cost table
        num_servers = len(server_lst)
        num_partitions = len(self.service_set.partitions)
        self.optimistic_cost_table = np.zeros(shape=(num_partitions, num_servers))
        for partition in self.service_set.partitions:
            if len(partition.predecessors) == 0:
                for s_idx in range(num_servers):
                    self.calc_optimistic_cost_table(partition, s_idx, server_lst)
        self.rank_oct = np.mean(self.optimistic_cost_table, axis=1)
        self.rank_oct_schedule = np.array(np.array(sorted(zip(self.rank_oct, np.arange(num_partitions)), reverse=True), dtype=np.int32)[:,1], dtype=np.int32)
        self.scheduling_policy = 'rank_oct'

    def constraint_chk(self, s_id=None):
        if s_id != None:
            # return self.server[s_id].constraint_chk()
            if len(np.where(self.deployed_server==s_id)[0]) > 0:
                server = self.server[s_id]
                idx = np.where(self.deployed_server==s_id)
                memory_consumption = sum(self.partition_memory_map[idx])
                
                # if s_id in self.local or s_id in self.request: # 임시
                #     node_weight = self.computation_time_table[idx, s_id].flatten() * self.partition_arrival_map[idx]
                #     edge_weight = dag_completion_time.get_edge_energy_weight(self.num_servers, self.num_servers-2, self.num_servers-1, s_id, self.net_manager.B_edge_up, self.net_manager.B_edge_down, self.net_manager.B_cloud_up, self.net_manager.B_cloud_down, self.net_manager.memory_bandwidth, self.deployed_server, self.service_set.input_data_size, self.net_manager.B_dd.flatten(), self.net_manager.P_d, self.partition_arrival_map)

                #     E_cp = sum(node_weight) * (server.max_energy_consumption - server.min_energy_consumption) * server.tau
                #     E_tr = edge_weight * server.tau
                #     energy_consumption = E_cp + E_tr
                # else:
                #     energy_consumption = 0.
                energy_consumption = 0.
                return (memory_consumption <= server.memory) and (energy_consumption <= server.cur_energy)
            else:
                return True
        else:
            # return [s.constraint_chk() for s in self.server.values()]
            constraint_chk = []
            for s_id in self.server.keys():
                if len(np.where(self.deployed_server==s_id)[0]) > 0:
                    server = self.server[s_id]
                    idx = np.where(self.deployed_server==s_id)
                    memory_consumption = sum(self.partition_memory_map[idx])
                    
                    # if s_id in self.local or s_id in self.request: # 임시
                    #     node_weight = self.computation_time_table[idx, s_id].flatten() * self.partition_arrival_map[idx]
                    #     edge_weight = dag_completion_time.get_edge_energy_weight(self.num_servers, self.num_servers-2, self.num_servers-1, s_id, self.net_manager.B_edge_up, self.net_manager.B_edge_down, self.net_manager.B_cloud_up, self.net_manager.B_cloud_down, self.net_manager.memory_bandwidth, self.deployed_server, self.service_set.input_data_size, self.net_manager.B_dd.flatten(), self.net_manager.P_d, self.partition_arrival_map)

                    #     E_cp = sum(node_weight) * (server.max_energy_consumption - server.min_energy_consumption) * server.tau
                    #     E_tr = edge_weight * server.tau
                    #     energy_consumption = E_cp + E_tr
                    # else:
                    #     energy_consumption = 0.
                    energy_consumption = 0.
                    constraint_chk.append((memory_consumption <= server.memory) and (energy_consumption <= server.cur_energy))
                else:
                    constraint_chk.append(True)
            return constraint_chk

    def set_service_set(self, service_set, arrival):
        self.service_set = service_set
        self.service_set.set_arrival(arrival)

    def after_timeslot(self, deployed_server, execution_order=None, timeslot=1):
        if self.num_timeslots > 1:
            self.set_env(deployed_server=deployed_server, execution_order=execution_order)
            self.service_set.update_arrival(timeslot)

            # energy update
            for server in self.server.values():
                server.energy_update()

    def total_time_dp(self):
        num_partitions = len(self.service_set.partitions)

        node_weight = np.array([self.computation_time_table[partition.total_id][self.deployed_server[partition.total_id]] for partition in self.service_set.partitions])
        edge_weight = dag_completion_time.get_edge_weight(self.num_servers, self.num_servers-2, self.num_servers-1, self.net_manager.B_edge_up, self.net_manager.B_edge_down, self.net_manager.B_cloud_up, self.net_manager.B_cloud_down, self.net_manager.memory_bandwidth, self.deployed_server, self.partition_device_map, self.service_set.input_data_size, self.net_manager.B_dd.flatten())

        self.finish_time = dag_completion_time.get_completion_time(num_partitions, self.deployed_server, self.execution_order, node_weight, edge_weight, self.service_set.partition_predecessor, self.service_set.partition_successor)

        result = np.zeros(len(self.service_set.services), dtype=np.float_)
        start = end = 0
        for svc in self.service_set.services:
            num_partitions = len(svc.partitions)
            start = end
            end += num_partitions
            result[svc.id] = max(self.finish_time[start:end])
        return result

    def get_completion_time_partition(self, p_id, finish_time=None, ready_time=None):
        num_partitions = len(self.service_set.partitions)
        if finish_time is None:
            finish_time = np.zeros(num_partitions)
        if ready_time is None:
            ready_time = np.zeros(num_partitions)
        if ready_time[p_id] == 0 and len(self.service_set.partition_predecessor[p_id]) == 0:
            ready_time[p_id] = self.net_manager.communication(self.service_set.input_data_size[(p_id,p_id)], self.partition_device_map[p_id], self.deployed_server[p_id])
        elif ready_time[p_id] == 0:
            TR_n = 0.
            for pred_id in self.service_set.partition_predecessor[p_id]:
                if finish_time[pred_id] > 0:
                    TF_p = finish_time[pred_id]
                else:
                    raise RuntimeError("Error: predecessor not finish, #{}".format(p_id, pred_id))
                T_tr = self.net_manager.communication(self.service_set.input_data_size[(pred_id,p_id)], self.deployed_server[pred_id], self.deployed_server[p_id])
                TR_n = max(TF_p + T_tr, TR_n)
            ready_time[p_id] = TR_n

        task_ready_time = max(ready_time[p_id], max(finish_time[np.where(self.deployed_server==self.deployed_server[p_id])]))
        finish_time[p_id] = task_ready_time + self.service_set.partitions[p_id].get_computation_time()
        return finish_time, ready_time

    def set_servers(self, request, local, edge, cloud):
        self.request = request
        self.local = local
        self.edge = edge
        self.cloud = cloud
        self.cloud_id = random.choice(list(cloud.keys()))
        self.server = {**request, **local, **edge, **cloud}
        self.computing_frequency = np.array([s.computing_frequency for s in self.server.values()])
        self.computing_intensity = np.array([s.computing_intensity for s in self.server.values()])
        self.computation_time_table = np.zeros(shape=(len(self.service_set.partitions), len(self.server)))
        for p in self.service_set.partitions:
            for s_id in self.server.keys():
                self.computation_time_table[p.total_id, s_id] = p.workload_size * self.computing_intensity[s_id][p.service.id] / self.computing_frequency[s_id]

    def init_env(self):
        for server in self.server.values():
            server.reset()
        deployed_server = np.full(shape=self.num_partitions, fill_value=self.cloud_id)
        self.set_env(deployed_server=deployed_server)

    def set_env(self, deployed_server, execution_order=None):
        [s.free() for s in self.server.values()]
        self.deployed_server = np.array(deployed_server, dtype=np.int32)
        start = end = 0
        for svc in self.service_set.services:
            start = end
            end += len(svc.partitions)
            for partition in svc.partitions:
                s_id = self.deployed_server[start + partition.id]
                partition.update(deployed_server=self.server[s_id])
                self.server[s_id].deploy_one(partition)

        if execution_order is not None:
            execution_order = np.array(execution_order, dtype=np.int32)
        else:
            # execution_order calculation, FCFS
            if self.scheduling_policy == 'rank_d':
                execution_order = self.rank_d_schedule

            # execution_order calculation, EFT
            elif self.scheduling_policy == 'rank_u':
                execution_order = self.rank_u_schedule

            # execution_order calculation, PEFT
            elif self.scheduling_policy == 'rank_oct':
                execution_order = self.rank_oct_schedule

        self.execution_order = np.zeros_like(execution_order)
        for idx, k in enumerate(execution_order):
            self.execution_order[k] = idx

    def get_reward(self):
        T_n = self.total_time_dp()
        # U_n = self.calc_utility(T_n)
        # print("T_n", T_n)
        # print("U_n", U_n)

        utility_factor = -np.max(T_n)

        # energy_factor = []
        # num_devices = 0
        # for d in list(self.request.values()) + list(self.local.values()) + list(self.edge.values()):
        #     E_d = d.energy_consumption()
        #     energy_factor.append(E_d)
        #     if E_d > 0:
        #         num_devices += 1
        # if num_devices == 0:
        #     energy_factor = 0
        # else:
        #     energy_factor = np.sum(energy_factor) #/ num_devices

        w_t = 1
        w_e = 0.000001
        reward = utility_factor * w_t #- energy_factor * w_e
        # print("energy_factor", energy_factor * w_e)
        # print("utility_factor", utility_factor * w_t)
        return reward

    def calc_utility(self, T_n):
        U_n = np.zeros(shape=(self.num_services, ))
        for n, svc in enumerate(self.service_set.services):
            T_n_hat = svc.deadline
            alpha = 5
            if T_n[n] < T_n_hat:
                U_n[n] = 1
            elif T_n_hat <= T_n[n] and T_n[n] < alpha * T_n_hat:
                U_n[n] = 1 - (T_n[n] - T_n_hat) / ((alpha - 1) * T_n_hat)
            else:
                U_n[n] = 0
        return U_n

    def calc_average(self):
        bandwidth = self.net_manager.B_dd
        self.average_bandwidth = np.mean(bandwidth[bandwidth < 1024*1024*1000])
        self.average_computing_power = np.mean(np.transpose([self.computing_frequency] * self.computing_intensity.shape[1]) / self.computing_intensity)

    def calc_rank_u_total_average(self, partition):    # rank_u for heft
        w_i = partition.workload_size / self.average_computing_power
        communication_cost = [0,]
        for succ in partition.successors:
            c_ij = partition.get_output_data_size(succ) / self.average_bandwidth
            if self.rank_u[succ.total_id] == 0:
                self.calc_rank_u_total_average(succ)
            communication_cost.append(c_ij + self.rank_u[succ.total_id])
        self.rank_u[partition.total_id] = w_i + max(communication_cost)

    def calc_rank_d_total_average(self, partition):    # rank_d for heft
        pred_rank = [0,]
        for pred in partition.predecessors:
            w_j = pred.workload_size / self.average_computing_power
            c_ij = partition.get_input_data_size(pred) / self.average_bandwidth
            if self.rank_d[pred.total_id] == 0:
                self.calc_rank_d_total_average(pred)
            pred_rank.append(w_j + c_ij + self.rank_d[pred.total_id])
        self.rank_d[partition.total_id] = max(pred_rank)

    def calc_optimistic_cost_table(self, partition, s_idx, server_lst):    # rank_oct for peft
        s_id = server_lst[s_idx]
        t_j_lst = [0,]
        for succ in partition.successors:
            p_w_lst = [np.inf]
            for succ_s_idx, succ_s_id in enumerate(server_lst):
                server = self.server[succ_s_id]
                w_jw = succ.workload_size * server.computing_intensity[succ.service.id] / server.computing_frequency
                c_ij = self.net_manager.communication(partition.get_output_data_size(succ), s_id, succ_s_id)
                oct_jw = self.optimistic_cost_table[succ.total_id, succ_s_idx]
                if oct_jw == 0:
                    self.calc_optimistic_cost_table(succ, succ_s_idx, server_lst)
                    oct_jw = self.optimistic_cost_table[succ.total_id, succ_s_idx]
                p_w_lst.append(oct_jw + w_jw + c_ij)
            t_j_lst.append(min(p_w_lst))
        self.optimistic_cost_table[partition.total_id, s_idx] = max(t_j_lst)


TYPE_CHAIN = 0
TYPE_PARALLEL = 1


class Partition:
    # the partition of the service in simulation
    def __init__(self, svc_set, service, **kwargs):
        self.id = None
        self.total_id = None
        self.service = service
        self.svc_set = svc_set
        self.deployed_server = None
        self.execution_order = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_input_data_size(self, pred):
        return self.service.input_data_size[(pred.id, self.id)]

    def get_output_data_size(self, succ):
        return self.service.output_data_size[(self.id, succ.id)]

    def get_deployed_server(self):
        return self.deployed_server

    def get_execution_order(self):
        return self.execution_order

    def reset(self):
        self.deployed_server = None
        self.execution_order = None

    def update(self, deployed_server=None, execution_order=None):
        if deployed_server != None:
            self.deployed_server = deployed_server
        if execution_order != None:
            self.execution_order = execution_order

    def set_successor_partition(self, partition):
        self.successors[partition.id] = partition

    def set_predecessor_partition(self, partition):
        self.predecessors[partition.id] = partition

    def get_computation_time(self): # for state
        T_cp = self.workload_size * self.deployed_server.computing_intensity[self.service.id] / self.deployed_server.computing_frequency
        return T_cp


class Service:
    #  the one of services in simulation
    def __init__(self, model_name, deadline):
        self.id = None
        self.partitions = list()
        self.model_name = model_name
        self.deadline = deadline
        self.arrival = None

        # for DAG completion time calculation
        self.num_partitions = None
        self.input_data_size = dict()
        self.output_data_size = dict()
        self.partition_computation_amount = list()
        self.partition_predecessor = dict()
        self.partition_successor = dict()

        self.deployed_server = None
        self.execution_order = None
        self.ready_time = None
        self.finish_time = None

    def update_successor(self, successor):
        for predecessor in successor.predecessors:
            if predecessor and successor not in predecessor.successors:
                predecessor.successors.append(successor)
            self.update_successor(predecessor)


class ServiceSet:
    #  set of services in simulation
    def __init__(self):
        self.services = list()   # the set of services
        self.partitions = list()   # the set of whole partitions
        self.input_data_size = dict()
        self.output_data_size = dict()
        self.partition_predecessor = dict()
        self.partition_successor = dict()
        self.service_arrival = None

    def add_services(self, svc):
        svc.id = len(self.services)
        self.services.append(svc)
        svc.num_partitions = len(svc.partitions)
        for id, partition in enumerate(svc.partitions):
            partition.id = id
            partition.total_id = len(self.partitions)
            partition.service = svc
            self.partitions.append(partition)

    def set_arrival(self, arrival):
        self.service_arrival = arrival
        for svc in self.services:
            svc.arrival = self.service_arrival[0, svc.id]

    def update_arrival(self, timeslot):
        for svc in self.services:
            svc.arrival = self.service_arrival[timeslot, svc.id]


class Server:
    #  elements with computing power
    def __init__(self, **kwargs):
        self.id = None
        self.deployed_partition = dict()
        self.deployed_partition_memory = dict()
        if not hasattr(self, "max_energy"):  # max battery
            self.max_energy = 10.
        if not hasattr(self, "cur_energy"):  # current battery
            self.cur_energy = 10.
        if not hasattr(self, "system_manager"):
            self.system_manager = None
        if not hasattr(self, "min_energy_consumption"):
            self.min_energy_consumption = 1.0
        if not hasattr(self, "max_energy_consumption"):
            self.max_energy_consumption = 10.0
        if not hasattr(self, "tau"):
            self.tau = 1 # sec

        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self):
        self.free()
        self.cur_energy = self.max_energy

    def free(self):
        while self.deployed_partition:
            _, partition = self.deployed_partition.popitem()
            partition.reset()
        if self.deployed_partition_memory:
            self.deployed_partition_memory.clear()

    def deploy_one(self, partition):
        self.deployed_partition[(partition.service.id, partition.id)] = partition
        self.deployed_partition_memory[(partition.service.id, partition.id)] = partition.memory

    def undeploy_one(self, partition):
        self.deployed_partition.pop((partition.service.id, partition.id))
        self.deployed_partition_memory.pop((partition.service.id, partition.id))

    def constraint_chk(self, *args):
        if len(self.deployed_partition) == 0:
            return True
        elif sum(self.deployed_partition_memory.values()) <= self.memory: # and self.energy_consumption() <= self.cur_energy: 임시
            return True
        else:
            # print("\tmemory", max(self.deployed_partition_memory.values(), default=0) <= self.memory)
            # print("\tenergy", self.energy_consumption() <= self.cur_energy)
            return False

    def energy_update(self):
        self.cur_energy -= self.energy_consumption()
        if self.cur_energy <= 0.:
            self.cur_energy = 0.
            self.free()

    def energy_consumption(self):
        # E_d = self.computation_energy_consumption() + self.transmission_energy_consumption()
        # return E_d
        if self.id in self.system_manager.local or self.id in self.system_manager.request:
            E_d = self.computation_energy_consumption() + self.transmission_energy_consumption()
            return E_d
        else:
            return 0

    def computation_energy_consumption(self):
        E_cp_d = 0.
        computation_energy = (self.max_energy_consumption - self.min_energy_consumption) * self.tau
        for c in self.deployed_partition.values():
            T_cp = c.get_computation_time()
            E_cp_d += T_cp * c.service.arrival
        E_cp_d *= computation_energy
        return E_cp_d

    def transmission_energy_consumption(self):
        E_tr_d = 0.
        for c in self.deployed_partition.values():
            E_tr_dnm = 0.
            for succ in c.successors:
                T_tr = self.system_manager.net_manager.communication(c.get_output_data_size(succ), self.system_manager.deployed_server[c.total_id], self.system_manager.deployed_server[succ.total_id])
                E_tr_dnm += self.system_manager.net_manager.P_d[self.system_manager.deployed_server[c.total_id]] * T_tr
            E_tr_d += E_tr_dnm * c.service.arrival * self.tau
        return E_tr_d