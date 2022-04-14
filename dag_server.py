from collections import deque
from os import times
import random
import math
from time import time
import numpy as np


COMP_RATIO = 50*(10**6) # computation per input data (50 MFLOPS)
MEM_RATIO = 1024 # memory usage per input data (1 KB)


class NetworkManager:  # managing data transfer
    def __init__(self, channel_bandwidth, channel_gain, gaussian_noise, B_edge, B_cloud, system_manager):
        self.C = channel_bandwidth
        self.g_wd = channel_gain
        self.sigma_w = gaussian_noise
        self.B_gw = B_edge
        self.B_edge = B_edge
        self.B_cloud = B_cloud
        self.B_dd = None
        self.P_dd = None
        self.system_manager = system_manager

    def communication(self, amount, sender, receiver):
        if sender == receiver:
            return 0
        elif sender in self.system_manager.local and receiver in self.system_manager.local:
            return amount / self.B_dd[sender, receiver]
        elif sender in self.system_manager.cloud or receiver in self.system_manager.cloud:
            return amount / self.B_cloud
        elif sender in self.system_manager.edge or receiver in self.system_manager.edge:
            return amount / self.B_edge

    def cal_b_dd(self):
        self.B_dd = np.zeros_like(self.P_dd)
        for i in range(self.B_dd.shape[0]):
            for j in range(i + 1, self.B_dd.shape[1]):
                SINR = self.g_wd * self.P_dd[i, j] / (self.sigma_w ** 2)
                self.B_dd[i, j] = self.B_dd[j, i] = self.C * math.log2(1 + SINR)
            self.B_dd[i, i] = float("inf")

    def get_b_dd(self):
        B_dd = np.zeros_like(self.B_dd)
        for i in range(B_dd.shape[0]):
            for j in range(B_dd.shape[1]):
                if self.B_dd[i, j] == float("inf"):
                    B_dd[i, j] = -1
                else:
                    B_dd[i, j] = self.B_dd[i, j]
        return B_dd


class SystemManager():
    def __init__(self):
        self.deployed_server = None
        self.execution_order = None
        self.cloud = None
        self.edge = None
        self.local = None
        self.server = None

        # service and partition info
        self.service_set = None
        
        # for network bandwidth and arrival rate
        self.net_manager = None
        self.service_arrival = None
        self.partition_arrival = None
        self.max_arrival = None

        self.num_servers = None
        self.num_services = None
        self.num_partitions = None
        self.cloud_id = None
        self.timeslot = 0
        self.rank_u = None
        self.rank_oct = None

    def constraint_chk(self, deployed_server, execution_order, s_id=None):
        self.set_env(deployed_server=deployed_server, execution_order=execution_order)
        if s_id != None:
            return self.server[s_id].constraint_chk()
        else:
            return [s.constraint_chk() for s in self.server.values()]

    def set_service_set(self, service_set, arrival, max_arrival):
        self.service_set = service_set
        self.service_arrival = arrival
        self.max_arrival = max_arrival
        self.partition_arrival = np.zeros(len(self.service_set.partitions), dtype=np.float_)
        for svc in service_set.services:
            for partition in svc.partitions:
                self.partition_arrival[partition.id] = self.service_arrival[0][svc.id]

    def after_timeslot(self, deployed_server, execution_order, timeslot):
        self.set_env(deployed_server=deployed_server, execution_order=execution_order)

        # energy update
        for server in self.server.values():
            server.energy_update()

        # arrival rate update
        for svc in self.service_set.services:
            for partition in svc.partitions:
                self.partition_arrival[partition.id] = self.service_arrival[timeslot][svc.id]

    def total_time(self):
        result = np.zeros(len(self.service_set.services), dtype=np.float_)
        for svc in self.service_set.services:
            result[svc.id] = svc.get_completion_time(self.net_manager)
        return result

    def get_completion_time(self, partition_id, Tr=None, Tf=None):
        return self.service_set.partitions[partition_id].service.get_completion_time_partition(self.net_manager, partition_id, Tr, Tf)

    def set_servers(self, local, edge, cloud):
        self.local = local
        self.edge = edge
        self.cloud = cloud
        self.cloud_id = random.choice(list(cloud.keys()))
        self.server = {**local, **edge, **cloud}

    def init_env(self):
        for server in self.server.values():
            server.reset()
        deployed_server = np.full(shape=self.num_partitions, fill_value=self.cloud_id)
        execution_order = np.arange(self.num_partitions)
        self.set_env(deployed_server=deployed_server, execution_order=execution_order)

    def set_env(self, deployed_server, execution_order):
        [s.free() for s in self.server.values()]
        self.deployed_server = np.array(deployed_server, dtype=np.int32)
        for partition in self.service_set.partitions:
            s_id = self.deployed_server[partition.id]
            partition.update(deployed_server=self.server[s_id])
            self.server[s_id].deploy_one(partition)
        self.execution_order = np.array(execution_order, dtype=np.int32)
        for order, p_id in enumerate(self.execution_order):
            self.service_set.partitions[p_id].update(execution_order=order)

    def get_state(self):
        next_state = []

        return np.array(next_state)

    def get_reward(self):
        T_n = self.total_time()
        U_n = self.calc_utility(T_n)
        #print("T_n", T_n)
        #print("U_n", U_n)

        utility_factor = sum(U_n)

        energy_factor = []
        for d in self.local.values():
            E_d = d.energy_consumption()
            energy_factor.append(E_d)
        energy_factor = 1 / np.sum(energy_factor)

        reward = utility_factor + energy_factor / 100
        #print("energy_factor", energy_factor)
        #print("utility_factor", utility_factor)
        return reward

    def calc_utility(self, T_n):
        U_n = np.zeros(shape=(self.num_services, ))
        for n, svc in enumerate(self.service_set.services):
            T_n_hat = svc.deadline
            alpha = 2
            if T_n[n] < T_n_hat:
                U_n[n] = 1
            elif T_n_hat <= T_n[n] and T_n[n] < alpha * T_n_hat:
                U_n[n] = 1 - (T_n[n] - T_n_hat) / ((alpha - 1) * T_n_hat)
            else:
                U_n[n] = 0
        return U_n

    def calc_average(self):
        bandwidth = self.net_manager.B_dd
        self.average_bandwidth = np.mean(bandwidth[bandwidth != np.inf])
        self.average_computing_power = np.mean([s.computing_frequency / s.computing_intensity for s in self.server.values()])

    def calc_rank_u(self, partition):    # rank_u for heft
        w_i = partition.workload_size / self.average_computing_power
        communication_cost = [0,]
        for succ in partition.successors:
            c_ij = partition.get_output_data_size(succ) / self.average_bandwidth
            if self.rank_u[succ.id] == 0:
                self.calc_rank_u(succ)
            communication_cost.append(c_ij + self.rank_u[succ.id])
        self.rank_u[partition.id] = w_i + max(communication_cost)

    def calc_rank_real(self, partition):    # rank_real for heft
        w_i = partition.workload_size / self.average_computing_power
        communication_cost = [0,]
        for succ in partition.successors:
            c_ij = partition.get_output_data_size(succ) / self.average_bandwidth
            if self.rank_u[succ.id] == 0:
                self.calc_rank_u(succ)
            communication_cost.append(c_ij + self.rank_u[succ.id])
        self.rank_u[partition.id] = w_i + max(communication_cost)

    def calc_rank_oct(self, partition):
        pass


TYPE_CHAIN = 0
TYPE_PARALLEL = 1


class Partition:
    # the partition of the service in simulation
    def __init__(self, service, **kwargs):
        self.id = None
        self.service = service
        self.deployed_server = None
        self.execution_order = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.total_predecessors = set()
        self.total_successors = set()
        self.total_pred_succ_id = None
    
    def find_total_predecessors(self):
        if len(self.successors) == 0:
            return
        for succ in self.successors:
            succ.total_predecessors.add(self)
            for pred in self.total_predecessors:
                succ.total_predecessors.add(pred)
            succ.find_total_predecessors()
    
    def find_total_successors(self):
        if len(self.predecessors) == 0:
            return
        for pred in self.predecessors:
            pred.total_successors.add(self)
            for succ in self.total_successors:
                pred.total_successors.add(succ)
            pred.find_total_successors()

    def get_input_data_size(self, pred):
        return self.service.input_data_size[(pred, self)]

    def get_output_data_size(self, succ):
        return self.service.output_data_size[(self, succ)]

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

    def get_task_ready_time(self, net_manager):
        if len(self.predecessors) == 0:
            TR_n = self.get_input_data_size(self) / net_manager.B_gw
        else:
            TR_n = 0
            for pred in self.predecessors:
                if self.service.finish_time[pred.id]:
                    TF_p = self.service.finish_time[pred.id]
                else:
                    TF_p = pred.get_task_finish_time(net_manager)
                    self.service.finish_time[pred.id] = TF_p
                T_tr = net_manager.communication(self.get_input_data_size(pred), self.deployed_server.id, pred.deployed_server.id)
                TR_n = max(TF_p + T_tr, TR_n)
        return TR_n

    def get_task_finish_time(self, net_manager):
        if self.service.ready_time[self.id] == 0: # for redundant calc
            self.service.ready_time[self.id] = self.get_task_ready_time(net_manager)
        TR_n = [self.service.ready_time[self.id]]

        for c in self.service.partitions:
            if c.deployed_server == self.deployed_server and c.execution_order < self.execution_order:
                if self.service.finish_time[c.id] > 0: # for redundant calc
                    finish_time = self.service.finish_time[c.id]
                    TR_n.append(finish_time)
                elif self.service.ready_time[c.id] > 0:
                    finish_time = c.get_task_finish_time(net_manager)
                    self.service.finish_time[c.id] = finish_time
                    TR_n.append(finish_time)
        T_cp = self.get_computation_time()
        self.service.finish_time[self.id] = max(TR_n) + T_cp
        return self.service.finish_time[self.id]

    def get_computation_time(self): # for state
        T_cp = self.workload_size * self.deployed_server.computing_intensity / self.deployed_server.computing_frequency
        return T_cp


class Service:
    #  the one of services in simulation
    def __init__(self, deadline):
        self.id = None
        self.partitions = list()
        self.deadline = deadline
        self.input_data_size = dict()
        self.output_data_size = dict()
        self.ready_time = None
        self.finish_time = None

    #  calculate the total completion time of the dag
    def get_completion_time(self, net_manager):
        # initialize
        self.ready_time = np.zeros(shape=len(self.partitions))
        self.finish_time = np.zeros(shape=len(self.partitions))
        for c in self.partitions:
            if len(c.predecessors) == 0:
                self.ready_time[c.id] = c.get_task_ready_time(net_manager)
        
        while 0 in self.finish_time:
            for c in self.partitions:
                if self.ready_time[c.id]:
                    self.finish_time[c.id] = c.get_task_finish_time(net_manager)
                    for succ in c.successors:
                        if 0 not in [self.finish_time[pred.id] for pred in succ.predecessors]:
                            self.ready_time[succ.id] = succ.get_task_ready_time(net_manager)
        return max(self.finish_time)

    #  calculate the total completion time of the dag
    def get_completion_time_partition(self, net_manager, partition_id, ready_time=None, finish_time=None):
        # initialize
        if ready_time is None:
            self.ready_time = np.zeros(shape=len(self.partitions))
        else:
            self.ready_time = ready_time
        if finish_time is None:
            self.finish_time = np.zeros(shape=len(self.partitions))
        else:
            self.finish_time = finish_time
        
        return self.partitions[partition_id].get_task_finish_time(net_manager), self.ready_time, self.finish_time

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

    def add_services(self, svc):
        svc.id = len(self.services)
        self.services.append(svc)
        for partition in svc.partitions:
            partition.id = len(self.partitions)
            partition.service = svc
            self.partitions.append(partition)


class Server:
    #  elements with computing power
    def __init__(self, computing_frequency, computing_intensity, memory, **kwargs):
        self.id = None
        self.computing_frequency = computing_frequency
        self.computing_intensity = computing_intensity
        self.memory = memory
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
        self.deployed_partition[partition.id] = partition
        self.deployed_partition_memory[partition.id] = partition.memory

    def undeploy_one(self, partition):
        self.deployed_partition.pop(partition.id)
        self.deployed_partition_memory.pop(partition.id)

    def constraint_chk(self, *args):
        if len(self.deployed_partition) == 0:
            return True
        elif max(self.deployed_partition_memory.values(), default=0) <= self.memory and self.energy_consumption() <= self.cur_energy:
            return True
        else:
            return False

    def energy_update(self):
        self.cur_energy -= self.energy_consumption()
        if self.cur_energy <= 0.:
            self.cur_energy = 0.
            self.free()

    def energy_consumption(self):
        if self.id in self.system_manager.local:
            E_d = self.computation_energy_consumption() + self.transmission_energy_consumption()
            return E_d
        else:
            return 0

    def computation_energy_consumption(self):
        E_cp_d = self.min_energy_consumption
        for c in self.deployed_partition.values():
            T_cp = c.get_computation_time()
            E_cp_d += (self.max_energy_consumption - self.min_energy_consumption) * T_cp * self.system_manager.partition_arrival[c.id]
        return E_cp_d

    def transmission_energy_consumption(self):
        E_tr_d = 0.
        for c in self.deployed_partition.values():
            E_tr_dnm = 0.
            for succ in c.successors:
                T_tr = self.system_manager.net_manager.communication(c.get_output_data_size(succ), self.system_manager.deployed_server[c.id], self.system_manager.deployed_server[succ.id])
                E_tr_dnm += self.system_manager.net_manager.P_dd[self.system_manager.deployed_server[c.id], self.system_manager.deployed_server[succ.id]] * T_tr
            E_tr_d += E_tr_dnm * self.system_manager.partition_arrival[c.id]
        return E_tr_d