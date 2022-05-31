import time
import random
import math
import numpy as np
from copy import deepcopy

import dag_completion_time


COMP_RATIO = 50*(10**6) # computation per input data (50 MFLOPS)
MEM_RATIO = 1024 # memory usage per input data (1 KB)


class NetworkManager:  # managing data transfer
    def __init__(self, channel_bandwidth, channel_gain, gaussian_noise, B_edge_up, B_edge_down, B_cloud_up, B_cloud_down, request, local, edge, cloud):
        self.C = channel_bandwidth
        self.g_wd = channel_gain
        self.sigma_w = gaussian_noise
        self.request_device = [0,1,2,3,4,5,6,7,8]
        self.B_edge_up = B_edge_up
        self.B_edge_down = B_edge_down
        self.B_cloud_up = B_cloud_up
        self.B_cloud_down = B_cloud_down
        self.B_dd = None
        self.P_dd = None

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
        self.allocated_resources = None
        self.cloud = None
        self.edge = None
        self.local = None
        self.request = None
        self.server = None

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
        self.rank_ready = None

    def constraint_chk(self, deployed_server, allocated_resources, s_id=None):
        self.set_env(deployed_server=deployed_server, allocated_resources=allocated_resources)
        if s_id != None:
            return self.server[s_id].constraint_chk()
        else:
            return [s.constraint_chk() for s in self.server.values()]

    def set_service_set(self, service_set, arrival):
        self.service_set = service_set
        self.service_set.set_arrival(arrival)

    def after_timeslot(self, deployed_server, allocated_resources, timeslot=1):
        if self.num_timeslots > 1:
            self.set_env(deployed_server=deployed_server, allocated_resources=allocated_resources)
            self.service_set.update_arrival(timeslot)

            # energy update
            for server in self.server.values():
                server.energy_update()

    def total_time(self):
        result = np.zeros(len(self.service_set.services), dtype=np.float_)
        start = end = 0
        for svc in self.service_set.services:
            num_partitions = len(svc.partitions)
            start = end
            end += num_partitions
            deployed_server = np.array(self.deployed_server[start:end], dtype=np.int32)
            allocated_resources = np.array(self.allocated_resources[start:end], dtype=np.int32)
            result[svc.id] = svc.get_completion_time(deployed_server, allocated_resources, self.num_servers, self.net_manager)
        return result

    def set_servers(self, request, local, edge, cloud):
        self.request = request
        self.local = local
        self.edge = edge
        self.cloud = cloud
        self.cloud_id = random.choice(list(cloud.keys()))
        self.server = {**request, **local, **edge, **cloud}

    def init_env(self):
        for server in self.server.values():
            server.reset()
        deployed_server = np.full(shape=self.num_partitions, fill_value=self.cloud_id)
        allocated_resources = np.full(shape=self.num_partitions, fill_value=1)
        self.set_env(deployed_server=deployed_server, allocated_resources=allocated_resources)

    def set_env(self, deployed_server, allocated_resources):
        [s.free() for s in self.server.values()]
        self.deployed_server = np.array(deployed_server, dtype=np.int32)
        self.allocated_resources = np.array(allocated_resources, dtype=np.float_)
        start = end = 0
        for svc in self.service_set.services:
            start = end
            end += len(svc.partitions)
            for partition in svc.partitions:
                s_id = self.deployed_server[start + partition.id]
                resources = self.allocated_resources[start + partition.id]
                partition.update(deployed_server=self.server[s_id], allocated_resources=resources)
                self.server[s_id].deploy_one(partition)

    def get_reward(self):
        T_n = self.total_time()
        # U_n = self.calc_utility(T_n)
        # print("T_n", T_n)
        # print("U_n", U_n)

        utility_factor = -sum(T_n)

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


TYPE_CHAIN = 0
TYPE_PARALLEL = 1


class Partition:
    # the partition of the service in simulation
    def __init__(self, service, **kwargs):
        self.id = None
        self.service = service
        self.deployed_server = None
        self.allocated_resources = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_input_data_size(self, pred):
        return self.service.input_data_size[(pred.id, self.id)]

    def get_output_data_size(self, succ):
        return self.service.output_data_size[(self.id, succ.id)]

    def get_deployed_server(self):
        return self.deployed_server

    def get_allocated_resources(self):
        return self.allocated_resources

    def reset(self):
        self.deployed_server = None
        self.allocated_resources = None

    def update(self, deployed_server=None, allocated_resources=None):
        if deployed_server != None:
            self.deployed_server = deployed_server
        if allocated_resources != None:
            self.allocated_resources = allocated_resources

    def set_successor_partition(self, partition):
        self.successors[partition.id] = partition

    def set_predecessor_partition(self, partition):
        self.predecessors[partition.id] = partition

    def get_computation_time(self): # for state
        T_cp = self.workload_size * self.deployed_server.computing_intensity[self.service.id] / self.deployed_server.computing_frequency / self.allocated_resources
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
        self.num_layers = None
        self.num_partitions = None
        self.layer_partition_map = None
        self.input_data_size = dict()
        self.output_data_size = dict()
        self.partition_predecessor = dict()
        self.partition_successor = dict()

    #  calculate the total completion time of the dag
    def get_completion_time(self, deployed_server, allocated_resources, num_servers, net_manager):
        node_weight = np.array([p.get_computation_time() for p in self.partitions])
        return dag_completion_time.get_completion_time_ra(self.num_partitions, self.num_layers, num_servers, num_servers-2, num_servers-1, net_manager.request_device[self.id], net_manager.B_edge_up, net_manager.B_edge_down, net_manager.B_cloud_up, net_manager.B_cloud_down, net_manager.memory_bandwidth, self.input_data_size, net_manager.B_dd.flatten(), deployed_server, node_weight, self.layer_partition_map)

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
        self.service_arrival = None

    def add_services(self, svc):
        svc.id = len(self.services)
        self.services.append(svc)
        svc.layer_partition_map = np.array([p.layer_idx for p in svc.partitions], dtype=np.int32)
        svc.num_layers = len(np.unique(svc.layer_partition_map))
        svc.num_partitions = len(svc.partitions)
        for id, partition in enumerate(svc.partitions):
            partition.id = id
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
        elif max(self.deployed_partition_memory.values(), default=0) <= self.memory and self.energy_consumption() <= self.cur_energy:
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
        E_cp_d = 0#self.min_energy_consumption
        for c in self.deployed_partition.values():
            T_cp = c.get_computation_time()
            E_cp_d += (self.max_energy_consumption - self.min_energy_consumption) * T_cp * c.service.arrival * self.tau
        return E_cp_d

    def transmission_energy_consumption(self):
        E_tr_d = 0.
        for c in self.deployed_partition.values():
            E_tr_dnm = 0.
            for succ in c.successors:
                T_tr = self.system_manager.net_manager.communication(c.get_output_data_size(succ), self.system_manager.deployed_server[c.id], self.system_manager.deployed_server[succ.id])
                E_tr_dnm += self.system_manager.net_manager.P_dd[self.system_manager.deployed_server[c.id], self.system_manager.deployed_server[succ.id]] * T_tr
            E_tr_d += E_tr_dnm * c.service.arrival * self.tau
        return E_tr_d