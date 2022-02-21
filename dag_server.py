from collections import deque
import random
import math
import numpy as np


COMP_RATIO = 50*(10**6) # computation per input data (50 MFLOPS)
MEM_RATIO = 1024 # memory usage per input data (1 KB)


class NetworkManager:  # managing data transfer
    def __init__(self, channel_bandwidth, channel_gain, gaussian_noise, system_manager):
        self.C = channel_bandwidth
        self.g_wd = channel_gain
        self.sigma_w = gaussian_noise
        self.B_gw = None
        self.B_fog = None
        self.B_cl = None
        self.B_dd = None
        self.P_dd = None
        self.system_manager = system_manager

    def communication(self, amount, sender, receiver):
        if sender == receiver:
            return 0
        elif sender in self.system_manager.edge and receiver in self.system_manager.edge:
            return amount / self.B_dd[sender, receiver]
        elif sender in self.system_manager.cloud or receiver in self.system_manager.cloud:
            return amount / self.B_cl
        elif sender in self.system_manager.fog or receiver in self.system_manager.fog:
            return amount / self.B_fog

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
        self.fog = None
        self.edge = None
        self.server = None

        # service and partition info
        self.service_set = None

        # for constraint check
        self.server_cpu = None
        self.server_mem = None
        self.partition_cpu = None
        self.partition_mem = None
        
        # for network bandwidth and arrival rate
        self.net_manager = None
        self.service_arrival = None
        self.partition_arrival = None
        self.max_arrival = None

        self.num_servers = None
        self.num_services = None
        self.num_partitions = None
        self.cloud_id = None
        self.ranku = None

    def constraint_chk(self, deployed_server, s_id=None):
        self.set_env(deployed_server=deployed_server)
        if s_id:
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
        self.partition_mem = np.zeros(len(self.service_set.partitions), dtype=np.int32)
        for partition in self.service_set.partitions:
            self.partition_mem[partition.id] = partition.memory

    def change_arrival(self, timeslot):
        for svc in self.service_set.services:
            for partition in svc.partitions:
                self.partition_arrival[partition.id] = self.service_arrival[timeslot][svc.id]

    def total_time(self):
        result = np.zeros(len(self.service_set.services), dtype=np.float_)
        for svc in self.service_set.services:
            result[svc.id] = svc.get_completion_time(self.net_manager)
        return result

    def get_completion_time(self, partition_id):
        return self.service_set.partitions[partition_id].get_completion_time(self.net_manager)

    def set_servers(self, edge, fog, cloud):
        self.edge = edge
        self.fog = fog
        self.cloud = cloud
        self.cloud_id = random.choice(list(cloud.keys()))
        self.server = {**edge, **fog, **cloud}

        self.server_cpu = np.zeros(self.num_servers)
        self.server_mem = np.zeros(self.num_servers)
        for id, s in self.server.items():
            self.server_cpu[id] = s.cpu
            self.server_mem[id] = s.memory

    def init_env(self, execution_order):
        deployed_server = np.full(shape=self.num_partitions, fill_value=self.cloud_id)
        self.set_env(deployed_server=deployed_server, execution_order=execution_order)

    def set_env(self, deployed_server=None, execution_order=None, cur_p_id=None, s_id=None):
        if cur_p_id != None and s_id != None:
            self.deployed_server[cur_p_id] = s_id
            deployed_server = self.deployed_server
        if deployed_server is not None:
            [s.reset() for s in self.server.values()]
            self.deployed_server = np.array(deployed_server, dtype=np.int32)
            for partition in self.service_set.partitions:
                s_id = self.deployed_server[partition.id]
                partition.update(deployed_server=self.server[s_id])
                self.server[s_id].deploy_one(partition)
        if execution_order is not None:
            self.execution_order = np.array(execution_order, dtype=np.int32)
            for order, p_id in enumerate(self.execution_order):
                self.service_set.partitions[p_id].update(execution_order=order)

    def get_state(self, next_p_id):
        temp = self.deployed_server[next_p_id]

        next_partition = self.service_set.partitions[next_p_id]
        next_state = [next_p_id]

        self.set_env(cur_p_id=next_p_id, s_id=self.cloud_id - 1)

        for s_id, s in self.server.items():
            if s_id != self.cloud_id:
                self.set_env(cur_p_id=next_p_id, s_id=s_id)
                TF = next_partition.get_completion_time(self.net_manager)
                T_cp = next_partition.computation_amount / s.cpu
                next_state += [
                    next_partition.get_ready_time(self.net_manager) * 1000, # task ready time
                    next_partition.get_computation_time() * 1000, # task computation time
                    sum(self.partition_mem[np.where(self.deployed_server == s_id)]) < s.memory, # server memory
                ]

        self.deployed_server[next_p_id] = temp
        return np.array(next_state)

    def get_reward(self, cur_p_id, timeslot):
        T_n = self.total_time()
        # U_n = self.calc_utility(T_n)
        # print("T_n", T_n)
        # print("U_n", U_n)

        utility_factor = 0
        for n in range(self.num_services):
            utility_factor += self.service_set.partitions[cur_p_id].get_completion_time(self.net_manager)

        energy_factor = []
        for d in self.edge.values():
            E_d = d.energy_consumption()
            E_d_hat = d.energy
            energy_factor.append(E_d_hat / E_d)
        energy_factor = np.mean(energy_factor)

        reward = utility_factor + energy_factor / 100
        #print("energy_factor", energy_factor)
        #print("utility_factor", utility_factor)

        if cur_p_id == -1 or cur_p_id == self.num_partitions - 1:
            return reward
        else:
            return 0

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
        self.average_bandwidth = np.mean(bandwidth[bandwidth > 0])
        self.average_computing_power = np.mean(self.server_cpu[:-1])

    def calc_ranku(self, partition):
        w_i = partition.computation_amount / self.average_computing_power
        communication_cost = [0,]
        for succ in partition.successors:
            c_ij = partition.get_output_data_size(succ) / self.average_bandwidth
            if self.ranku[succ.id] == 0:
                self.calc_ranku(succ)
            communication_cost.append(c_ij + self.ranku[succ.id])
        self.ranku[partition.id] = w_i + max(communication_cost)


TYPE_CHAIN = 0
TYPE_PARALLEL = 1


class Partition:
    # the partition of the service in simulation
    def __init__(self, service, computation_amount, memory):
        self.id = None
        self.deployed_server = None
        self.execution_order = None
        self.service = service
        self.computation_amount = computation_amount
        self.memory = memory
        self.successors = list()
        self.predecessors = list()

    def get_input_data_size(self, pred):
        return self.service.input_data_array[pred.id, self.id]

    def get_output_data_size(self, succ):
        return self.service.input_data_array[self.id, succ.id]

    def get_deployed_server(self):
        return self.deployed_server

    def get_execution_order(self):
        return self.execution_order

    def reset(self):
        self.deployed_server = None
        #self.execution_order = None

    def update(self, deployed_server=None, execution_order=None):
        if deployed_server != None:
            self.deployed_server = deployed_server
        if execution_order != None:
            self.execution_order = execution_order

    def set_successor_partition(self, partition):
        self.successors[partition.id] = partition

    def set_predecessor_partition(self, partition):
        self.predecessors[partition.id] = partition
    
    def get_completion_time(self, net_manager):
        self.service.finish_time = np.zeros(len(self.service.partitions))
        return self.get_task_finish_time(net_manager)
    
    def get_ready_time(self, net_manager):
        self.service.finish_time = np.zeros(len(self.service.partitions))
        return self.get_task_finish_time(net_manager) - self.get_computation_time()

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
        TR_n = [self.get_task_ready_time(net_manager)]
        for c in self.service.partitions:
            if c.deployed_server == self.deployed_server and c.execution_order < self.execution_order:
                if self.service.finish_time[c.id]:
                    finish_time = self.service.finish_time[c.id]
                else:
                    finish_time = c.get_task_finish_time(net_manager)
                    self.service.finish_time[c.id] = finish_time
                TR_n.append(finish_time)
        T_cp = self.get_computation_time()
        return max(TR_n) + T_cp

    def get_computation_time(self): # for state
        T_cp = self.computation_amount / self.deployed_server.cpu
        return T_cp


class Service:
    #  the one of services in simulation
    def __init__(self, deadline):
        self.id = None
        self.partitions = list()
        self.deadline = deadline
        self.input_data_array = None

    def calc_service_size(self, shape):
        if type(shape[1]) == int:
            return shape[1]
        elif type(shape[1]) in (tuple, list):
            size = 0
            for s in shape[1]:
                size += self.calc_service_size(shape=s)
            return size

    #  calculate the total completion time of the dag
    def get_completion_time(self, net_manager):
        return self.partitions[-1].get_completion_time(net_manager)

    # for the datagenerator, create dag reculsively
    def create_partitions(self, opt=(1, ((0, 3), (0, 4)))):
        #print("partition", opt)
        input_data_size = random.randint(64, 512) # 64KB~1MB
        computation_amount = input_data_size * COMP_RATIO
        memory = input_data_size * MEM_RATIO
        first_partition = Partition(self, computation_amount, memory)
        first_partition.id = len(self.partitions)
        self.input_data_array[first_partition.id, first_partition.id] = input_data_size
        self.partitions.append(first_partition)

        if opt[0] == TYPE_CHAIN:
            self.create_chain_partition(first_partition, opt[1])
        elif opt[0] == TYPE_PARALLEL:
            self.create_parallel_partition(first_partition, opt[1])
        else:
            raise RuntimeError('Unknown opt type!!')

        self.update_successor(self.partitions[-1])


    def create_chain_partition(self, predecessor, opt):
        #print("chain opt", opt)
        if type(opt) == int:
            for _ in range(opt):
                input_data_size = random.randint(64, 512) # 64KB~1MB
                computation_amount = input_data_size * COMP_RATIO
                memory = input_data_size * MEM_RATIO
                partition = Partition(self, computation_amount, memory)
                partition.id = len(self.partitions)
                if type(predecessor) in (tuple, list):
                    portion = np.array([p.computation_amount for p in predecessor])
                    portion = portion / sum(portion)
                    for i, p in enumerate(predecessor):
                        self.input_data_array[p.id, partition.id] = int(input_data_size * portion[i])
                        partition.predecessors.append(p)
                else:
                    self.input_data_array[predecessor.id, partition.id] = input_data_size
                    partition.predecessors.append(predecessor)
                self.partitions.append(partition)
                predecessor = partition
                #print("chain", predecessor.id)
            #print("chain return", predecessor.id)
            #print("chain return", predecessor)
            return predecessor

        elif type(opt) in (tuple, list):
            for element in opt:
                #print("chain element", element)
                if element[0] == TYPE_CHAIN:
                    predecessor = self.create_chain_partition(predecessor, element[1])
                elif element[0] == TYPE_PARALLEL:
                    predecessor = self.create_parallel_partition(predecessor, element[1])
                else:
                    raise RuntimeError('Unknown opt type!!')
            return predecessor

    def create_parallel_partition(self, predecessor, opt):
        #print("parallel opt", opt)
        if type(opt) == int:
            parallel_partitions = list()
            for _ in range(opt):
                input_data_size = random.randint(64, 512) # 64KB~1MB
                computation_amount = input_data_size * COMP_RATIO
                memory = input_data_size * MEM_RATIO
                partition = Partition(self, computation_amount, memory)
                partition.id = len(self.partitions)
                if type(predecessor) in (tuple, list):
                    portion = np.array([p.computation_amount for p in predecessor])
                    portion = portion / sum(portion)
                    for i, p in enumerate(predecessor):
                        self.input_data_array[p.id, partition.id] = int(input_data_size * portion[i])
                        partition.predecessors.append(p)
                else:
                    self.input_data_array[predecessor.id, partition.id] = input_data_size
                    partition.predecessors.append(predecessor)
                self.partitions.append(partition)
                parallel_partitions.append(partition)
            #print("parallel return", len(parallel_partitions))
            #print("parallel return", parallel_partitions)
            return parallel_partitions

        elif type(opt) in (tuple, list):
            parallel_partitions = list()
            for element in opt:
                #print("parallel element", element)
                if element[0] == TYPE_CHAIN:
                    parallel_partitions.append(self.create_chain_partition(predecessor, element[1]))
                elif element[0] == TYPE_PARALLEL:
                    parallel_partitions.append(self.create_parallel_partition(predecessor, element[1]))
                else:
                    raise RuntimeError('Unknown opt type!!')
            return parallel_partitions

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
    def __init__(self, cpu, memory, ipc, **kwargs):
        self.id = None
        self.cpu = cpu * ipc
        self.cpu_clk = cpu
        self.memory = memory
        self.ipc = ipc
        self.deployed_partition = dict()
        self.deployed_partition_mem = dict()
        if not hasattr(self, "energy"):  # battery
            self.energy = 100.
        if not hasattr(self, "tau"):   # timeslot
            self.tau = 60. * 60.    # 1hour default
        if not hasattr(self, "capacitance"):  # capacitance value (need reference)
            self.capacitance = 1e-6
        if not hasattr(self, "system_manager"):
            self.system_manager = None
        if not hasattr(self, "min_energy"):
            self.min_energy = 1.0
        if not hasattr(self, "max_energy"):
            self.max_energy = 10.0

        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self):
        while self.deployed_partition:
            _, partition = self.deployed_partition.popitem()
            partition.reset()
        if self.deployed_partition_mem:
            self.deployed_partition_mem.clear()

    def deploy_one(self, partition):
        self.deployed_partition[partition.id] = partition
        self.deployed_partition_mem[partition.id] = partition.memory

    def undeploy_one(self, partition):
        self.deployed_partition.pop(partition.id)
        self.deployed_partition_mem.pop(partition.id)

    def constraint_chk(self, *args):
        if sum(self.deployed_partition_mem.values()) <= self.memory and self.energy_consumption() <= self.energy:
            return True
        else:
            return False

    def energy_update(self):
        self.energy -= self.energy_consumption()
        if self.energy <= 0.:
            self.energy = 0.
            self.reset()

    def energy_consumption(self):
        if self.id in self.system_manager.edge:
            return self.computation_energy_consumption() + self.transmission_energy_consumption()
        else:
            return 0

    def computation_energy_consumption(self):
        E_cp_d = self.min_energy + (self.max_energy - self.min_energy)
        return E_cp_d

    def transmission_energy_consumption(self):
        E_tr_d = 0.
        for c in self.system_manager.service_set.partitions:
            E_tr_dnm = 0.
            if self.system_manager.deployed_server[c.id] in self.system_manager.edge:
                for succ in c.successors:
                    T_tr = self.system_manager.net_manager.communication(c.get_output_data_size(succ), self.system_manager.deployed_server[c.id], self.system_manager.deployed_server[succ.id])
                    E_tr_dnm += self.system_manager.net_manager.P_dd[self.system_manager.deployed_server[c.id], self.system_manager.deployed_server[succ.id]] * T_tr
            E_tr_d += E_tr_dnm * self.system_manager.partition_arrival[c.id]
        return E_tr_d




if __name__=="__main__":
    service_set = ServiceSet()
    num_services = 1
    deadline_opt = (10, 100)
    max_partitions = 5
    for i in range(num_services):
        # create service
        deadline = random.uniform(deadline_opt[0], deadline_opt[1])
        svc = Service(deadline)

        # create partitions
        dag_shape = (0, ((1, random.randint(2, max_partitions)), (0, random.randint(1, max_partitions)), (1, ((0, random.randint(1, max_partitions)), (0, random.randint(1, max_partitions)))), (0, 1)))
        dag_size = svc.calc_service_size(shape=dag_shape) + 1
        print("dag_shape", dag_shape)
        print("dag_size", dag_size)
        input()
        svc.input_data_array = np.zeros(shape=(dag_size, dag_size), dtype=np.int32)
        svc.create_partitions(opt=dag_shape)

        service_set.add_services(svc)

    for svc in service_set.services:
        for partition in svc.partitions:
            print("-------------------------")
            print("partition", partition.id)
            predecessors = list()
            successors = list()
            for p in partition.predecessors:
                predecessors.append(p.id)
            for p in partition.successors:
                successors.append(p.id)
            print("predecessor", predecessors)
            print("input data", [svc.input_data_array[p, partition.id] for p in predecessors])
            print("successor", successors)
        print("input_data_array", svc.input_data_array)
        input()