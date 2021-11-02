from collections import deque
import random
import math
import numpy as np


COMP_RATIO = 50*(10**6) # computation per input data (50 MFLOPS)
MEM_RATIO = 1024 # memory usage per input data (1 KB)


class NetworkManager:  # managing data transfer
    def __init__(self, channel_bandwidth, channel_gain, gaussian_noise):
        self.C = channel_bandwidth
        self.g_wd = channel_gain
        self.sigma_w = gaussian_noise
        self.B_fog = None
        self.B_cl = None
        self.B_dd = None
        self.P_dd = None

    def communication(self, amount, sender, receiver, system_manager):
        if sender == receiver:
            return 0
        elif sender in system_manager.edge and receiver in system_manager.edge:
            return amount / self.B_dd[sender, receiver]
        elif receiver in system_manager.edge and sender in system_manager.fog:
            return amount / self.B_fog
        elif receiver in system_manager.fog and sender in system_manager.edge:
            return amount / self.B_fog
        elif receiver in system_manager.fog and sender in system_manager.fog:
            return amount / self.B_fog
        elif receiver in system_manager.cloud or sender in system_manager.cloud:
            return amount / self.B_cl

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
        self._x = None
        self._y = None
        self.cloud = None
        self.fog = None
        self.edge = None
        self.server = None
        self.cpu_resource = None
        self.mem_resource = None
        self.service_set = None
        self.net_manager = None
        self.service_arrival = None
        self.container_arrival = None
        self.num_servers = None
        self.num_containers = None
        self.cloud_id = None

        self.NUM_CHANNEL = None
        # 0: which container which server
        # 1: deployed container computation amount
        # 2: deployed container memory
        # 3: deployed container arrival rate
        # 4: server cpu
        # 5: server memory
        # 6: server energy
        # 7~: dependency between containers & bandwidth between servers

    def set_service_set(self, service_set, arrival):
        self.service_set = service_set
        self.service_arrival = arrival
        self.container_arrival = np.zeros(len(self.service_set.container_set), dtype=np.float_)
        for svc in service_set.svc_set:
            for container in svc.partitions:
                self.container_arrival[container.id] = self.service_arrival[0][svc.id]

    def change_arrival(self, timeslot):
        for svc in self.service_set.svc_set:
            for container in svc.partitions:
                self.container_arrival[container.id] = self.service_arrival[timeslot][svc.id]

    def get_next_state(self, mat_y):
        self.set_y_mat(mat_y)
        next_state = np.zeros((self.NUM_CHANNEL, self.num_servers, self.num_containers))
        for c_id, s_id in enumerate(self._y):
            # 0: which container which server
            next_state[0,s_id,c_id] = 1
            # 1: deployed container computation amount
            next_state[1,:,c_id] = self.service_set.container_set[c_id].computation_amount / (10**12)
            # 2: deployed container memory
            next_state[2,:,c_id] = self.service_set.container_set[c_id].memory / (1024 * 1024 * 1024)
            # 3: deployed container arrival rate
            next_state[3,:,c_id] = self.container_arrival[c_id] / 50
        for s_id in range(self.num_servers):
            # 4: server remaining cpu
            next_state[4,s_id,:] = (self.cpu_resource[s_id] - self.server[s_id].used_cpu) / (10**12)
            # 5: server remaining memory
            next_state[5,s_id,:] = (self.mem_resource[s_id] - self.server[s_id].used_mem) / (1024 * 1024 * 1024)
            # 6: server remaining energy
            next_state[6,s_id,:] = (self.server[s_id].get_energy() - self.server[s_id].energy_consumption()) / 100
        for c_id, s_id in enumerate(self._y): # from deployed containers
            for container in range(self.num_containers): # this container have this transmission delay
                # 7~: dependency between containers & bandwidth between servers
                if self.service_set.container_set[c_id] in self.service_set.container_set[container].predecessors:
                    next_state[7+container,s_id,c_id] = self.net_manager.communication(self.service_set.container_set[c_id].output_data_size, self.service_set.container_set[c_id]._y, self.service_set.container_set[container]._y, self)
                if self.service_set.container_set[c_id] in self.service_set.container_set[container].successors:
                    next_state[7+container,s_id,c_id] = self.net_manager.communication(self.service_set.container_set[container].output_data_size, self.service_set.container_set[container]._y, self.service_set.container_set[c_id]._y, self)
        return next_state

    def total_time(self):  # use  todo: add transmission
        result = np.zeros(len(self.service_set.svc_set), dtype=np.float_)
        for svc in self.service_set.svc_set:
            result[svc.id] = svc.complete_time(self, self.net_manager)
        return result

    def set_servers(self, edge, fog, cloud):
        self.edge = edge
        self.fog = fog
        self.cloud = cloud
        self.cloud_id = random.choice(list(cloud.keys()))
        self.server = {**edge, **fog, **cloud}
        self.cpu_resource = np.zeros(len(self.server))
        self.mem_resource = np.zeros(len(self.server))
        for id, s in self.server.items():
            self.cpu_resource[id] = s.cpu
            self.mem_resource[id] = s.memory

    def init_servers(self, X):
        Y = np.full_like(X, list(self.cloud.keys())[0])
        self.set_xy_mat(X, Y)
        return Y

    def set_xy_mat(self, mat_x, mat_y):
        [s.reset() for id, s in self.server.items()]
        self._x = np.array(mat_x, dtype=np.float64)
        self._y = np.array(mat_y, dtype=np.int32)
        for c_id, s_id in enumerate(self._y):
            self.server[int(s_id)].deploy_one(self.service_set.container_set[c_id], self._x[c_id])

    def set_y_mat(self, mat_y):
        [s.reset() for id, s in self.server.items()]
        self._y = np.array(mat_y, dtype=np.int32)
        for c_id, s_id in enumerate(self._y):
            self.server[int(s_id)].deploy_one(self.service_set.container_set[c_id], self._x[c_id])

    def set_y(self, c_id, y):
        self._y[c_id] = y
        container = self.service_set.container_set[c_id]
        self.server[y].deploy_one(container, self._x[c_id])

    def reset_y(self, c_id, y, cloud_id):
        self._y[c_id] = cloud_id
        container = self.service_set.container_set[c_id]
        self.server[y].undeploy_one(container, self._x[c_id], cloud_id)


TYPE_CHAIN = 0
TYPE_PARALLEL = 1


class Container:
    # the partition of the service in simulation
    def __init__(self, input_data_size, output_data_size, computation_amount, memory):
        self.id = None
        self._x = None
        self._y = None
        self.input_data_size = input_data_size
        self.output_data_size = output_data_size
        self.computation_amount = computation_amount
        self.memory = memory
        self.successors = list()
        self.predecessors = list()
        self.svc = None

    def reset_xy(self):
        self._x = None
        self._y = None

    def update_xy(self, x=None, y=None):
        self._y = y
        self._x = x

    def service_rate(self):
        return self._x / self.computation_amount

    def set_successor_container(self, container):
        self.successors[container.id] = container

    def set_predecessor_container(self, container):
        self.predecessors[container.id] = container

    def get_task_ready_time(self, system_manager, net_manager):
        if len(self.predecessors) == 0:
            TR_m = self.input_data_size / net_manager.B_gw
        else:
            TR_m = 0
            for p in self.predecessors:
                TF_p = p.get_task_finish_time(system_manager, net_manager)
                T_tr = net_manager.communication(p.output_data_size, self._y, p._y, system_manager)
                TR_m = max(TF_p + T_tr, TR_m)
        return TR_m

    def get_task_finish_time(self, system_manager, net_manager):
        TR_m = self.get_task_ready_time(system_manager, net_manager)
        T_cp = self.get_computaion_time(system_manager)
        return TR_m + T_cp

    def get_computaion_time(self, system_manager): # for state
        if self._x is not None and self._x > 0:
            T_cp = self.computation_amount / self._x
        else:
            T_cp = float("inf")
        # cloud disadvantage
        if not self._y is system_manager.cloud_id:
            T_cp /= 10
        return T_cp

    def get_transmission_time(self, system_manager, net_manager): # for state
        if len(self.predecessors) == 0:
            T_tr = self.input_data_size / net_manager.B_gw
        else:
            T_tr = 0
            for p in self.predecessors:
                temp = net_manager.communication(p.output_data_size, self._y, p._y, system_manager)
                T_tr = max(T_tr, temp)
        return T_tr


class Service:
    #  the one of services in simulation
    def __init__(self, deadline):
        self.id = None
        self.partitions = list()
        self.deadline = deadline

    #  calculate the total completion time of the dag
    def complete_time(self, system_manager, net_manager):
        return self.partitions[-1].get_task_finish_time(system_manager, net_manager)

    # for the datagenerator, create dag reculsively
    def create_partitions(self, opt=(1, ((0, 3), (0, 4)))):
        #print("partition", opt)
        input_data_size = random.randint(256, 1024) # 64KB~1MB
        output_data_size = random.randint(256, 1024) # 64KB~1MB
        computation_amount = input_data_size * COMP_RATIO
        memory = input_data_size * MEM_RATIO
        first_partition = Container(input_data_size, output_data_size, computation_amount, memory)
        first_partition.id = len(self.partitions)
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
                input_data_size = random.randint(64, 1024) # 64KB~1MB
                output_data_size = random.randint(64, 1024) # 64KB~1MB
                computation_amount = input_data_size * COMP_RATIO
                memory = input_data_size * MEM_RATIO
                partition = Container(input_data_size, output_data_size, computation_amount, memory)
                partition.id = len(self.partitions)
                if type(predecessor) in (tuple, list):
                    for p in predecessor:
                        partition.predecessors.append(p)
                else:
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
                input_data_size = random.randint(64, 1024) # 64KB~1MB
                output_data_size = random.randint(64, 1024) # 64KB~1MB
                computation_amount = input_data_size * COMP_RATIO
                memory = input_data_size * MEM_RATIO
                partition = Container(input_data_size, output_data_size, computation_amount, memory)
                partition.id = len(self.partitions)
                if type(predecessor) in (tuple, list):
                    for p in predecessor:
                        partition.predecessors.append(p)
                else:
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
        self.svc_set = list()   # the set of services
        self.container_set = list()   # the set of whole containers

    def add_services(self, svc):
        svc.id = len(self.svc_set)
        self.svc_set.append(svc)
        for partition in svc.partitions:
            partition.id = len(self.container_set)
            partition.svc = svc
            self.container_set.append(partition)


class Server:
    #  elements with computing power
    def __init__(self, cpu, memory, ipc, **kwargs):
        self.id = None
        self.cpu = cpu * ipc
        self.cpu_clk = cpu
        self.memory = memory
        self.ipc = ipc
        self.deployed_container = dict()
        self.deployed_container_cpu = dict()
        self.deployed_container_mem = dict()
        self.used_cpu = 0.
        self.used_mem = 0.
        self.svc_set = None
        if not hasattr(self, "_energy"):  # battery
            self._energy = 100.
        if not hasattr(self, "_tau"):   # timeslot
            self._tau = 60. * 60.    # 1hour default
        if not hasattr(self, "_capacitance"):  # capacitance value (need reference)
            self._capacitance = 1e-6
        if not hasattr(self, "system_manager"):
            self.system_manager = None
        if not hasattr(self, "min_energy"):
            self.min_energy = 1.0
        if not hasattr(self, "max_energy"):
            self.max_energy = 10.0
        self.__trans_energy_ratio = 10.  # need reference

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.pushed_lst = deque()

    def computing_time(self, container):
        cpu = self.deployed_container_cpu.get(container.id)
        if cpu and cpu > 0.:
            return container.computation_amount / self.deployed_container_cpu[container.id]
        else:
            return -1.0

    def reset(self):
        self.used_cpu = self.used_mem = 0
        self.pushed_lst.clear()
        while self.deployed_container:
            _, container = self.deployed_container.popitem()
            container.reset_xy()
        if self.deployed_container_cpu:
            self.deployed_container_cpu.clear()
        if self.deployed_container_mem:
            self.deployed_container_mem.clear()

    def deploy_one(self, container, computing_resource):
        self.deployed_container[container.id] = container
        self.deployed_container_cpu[container.id] = computing_resource
        self.deployed_container_mem[container.id] = container.memory
        self.used_cpu += computing_resource
        self.used_mem += container.memory
        container.update_xy(computing_resource, self.id)

    def undeploy_one(self, container, computing_resource, cloud_id):
        self.deployed_container.pop(container.id)
        self.deployed_container_cpu.pop(container.id)
        self.deployed_container_mem.pop(container.id)
        self.used_cpu = sum(self.deployed_container_cpu.values())
        self.used_mem = sum(self.deployed_container_mem.values())
        container.update_xy(computing_resource, cloud_id)

    def constraint_chk(self, *args):
        if self.used_cpu <= self.cpu and self.used_mem <= self.memory and self.energy_consumption() <= self._energy:
            return np.inf
        else:
            return -np.inf

    def energy_update(self):
        self._energy -= self.energy_consumption()
        if self._energy <= 0.:
            self._energy = 0.
            self.reset()

    def get_energy(self):
        return self._energy

    def energy_consumption(self):
        if self.id in self.system_manager.edge:
            return self.computation_energy_consumption() + self.transmission_energy_consumption()
        else:
            return 0

    def computation_energy_consumption(self):
        return self.min_energy + (self.max_energy - self.min_energy) * self.used_cpu / self.cpu

    def transmission_energy_consumption(self):
        E_tr_d = 0.
        for c in self.system_manager.service_set.container_set:
            E_tr_dnm = 0.
            if self.system_manager._y[c.id] in self.system_manager.edge:
                for s in c.successors:
                    T_tr = self.system_manager.net_manager.communication(c.output_data_size, self.system_manager._y[c.id], self.system_manager._y[s.id], self.system_manager)
                    E_tr_dnm += self.system_manager.net_manager.P_dd[self.system_manager._y[c.id], self.system_manager._y[s.id]] * T_tr
            E_tr_d += E_tr_dnm * self.system_manager.container_arrival[c.id]
        return E_tr_d / 10 # for test

if __name__=="__main__":
    svc_set = ServiceSet()
    num_services = 10
    deadline_opt = (10, 100)
    max_partitions = 5
    for i in range(num_services):
        # create service
        deadline = random.uniform(deadline_opt[0], deadline_opt[1])
        svc = Service(deadline)

        # create partitions
        num_partitions = random.randint(1, max_partitions)
        svc.create_partitions(opt=(0, ((1, num_partitions), (0, num_partitions + 2), (1, ((0, num_partitions), (0, num_partitions))), (0, 1))))

        svc_set.add_services(svc)
    '''
    for svc in svc_set.svc_set:
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
            print("successor", successors)
        input()
    '''